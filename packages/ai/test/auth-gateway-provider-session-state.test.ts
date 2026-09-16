/**
 * The auth-gateway owns `providerSessionState` per logical session.
 *
 * Providers learn sticky lessons about an endpoint from rejections: Anthropic's
 * `fastModeDisabled` / `strictToolsDisabled` / `replayUnsignedThinkingDisabled`
 * flags, OpenAI's strict-tools and reasoning-effort fallbacks. The map holding
 * them is non-serializable, so `pi-native-client` strips it from the wire and
 * `pi-native-server` refuses it — a gateway client cannot supply one. Without a
 * server-side owner, every containerized / robomp turn re-pays the rejected
 * upstream round-trip that already taught the lesson.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { clearCustomApis } from "@oh-my-pi/pi-ai/api-registry";
import { AuthGatewaySessionStateStore, startAuthGateway } from "@oh-my-pi/pi-ai/auth-gateway";
import type { AuthGatewayServerHandle, AuthGatewaySessionStateRequest } from "@oh-my-pi/pi-ai/auth-gateway";
import { AuthStorage, type OAuthCredential } from "@oh-my-pi/pi-ai/auth-storage";
import * as oauthUtils from "@oh-my-pi/pi-ai/registry/oauth";
import { createMockModel, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import type { Api, Context, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { withOfficialAnthropicEndpoint } from "./helpers";

function makeAnthropicModel(baseUrl: string): Model<"anthropic-messages"> {
	return buildModel({
		id: "claude-opus-4-7",
		name: "claude-opus-4-7",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl,
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 8_192,
	});
}

const ANTHROPIC_MODEL: Model<"anthropic-messages"> = makeAnthropicModel("https://api.anthropic.com");

const CONTEXT: Context = {
	systemPrompt: ["Stay concise."],
	messages: [{ role: "user", content: "Hi", timestamp: 1 }],
};

const SSE_EVENTS: Array<Record<string, unknown>> = [
	{
		type: "message_start",
		message: {
			id: "msg_gateway_session_state",
			type: "message",
			role: "assistant",
			model: ANTHROPIC_MODEL.id,
			content: [],
			usage: { input_tokens: 1, output_tokens: 0 },
		},
	},
	{ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
	{ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
	{ type: "content_block_stop", index: 0 },
	{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
	{ type: "message_stop" },
];

/**
 * What the real API answers a `speed: "fast"` request with when the account or
 * model lacks the entitlement. Must stay classifiable as
 * `FastModeUnsupported`: HTTP 400 + `invalid_request_error` + `speed` +
 * "not support".
 */
const FAST_MODE_REJECTION = {
	type: "error",
	error: {
		type: "invalid_request_error",
		message: "speed: this model does not support fast mode for your account",
	},
};

/**
 * What the real API answers when the compiled tool grammar is too large. Must
 * stay classifiable as `Grammar`: HTTP 400 + `invalid_request_error` +
 * "compiled grammar" + "too large".
 */
const GRAMMAR_REJECTION = {
	type: "error",
	error: {
		type: "invalid_request_error",
		message:
			"The compiled grammar is too large, which would cause performance issues. Simplify your tool schemas or reduce the number of strict tools.",
	},
};

/**
 * What the real API answers once an account's own quota window is spent. Must
 * stay classifiable as a usage limit so the gateway blocks that credential and
 * rotates to a sibling instead of surfacing the failure.
 */
const USAGE_LIMIT_REJECTION = {
	type: "error",
	error: { type: "rate_limit_error", message: "usage_limit_reached" },
};

interface UpstreamPayload {
	speed?: string;
	tools?: Array<{ strict?: boolean }>;
}

interface Upstream {
	/** Base URL an `anthropic-messages` model can be pointed at. */
	url: string;
	/** One entry per upstream Anthropic request, in order. */
	payloads: UpstreamPayload[];
	/** The bearer each of those requests arrived with, same order. */
	bearers: Array<string | undefined>;
	/** Bearers whose account holds the fast-mode entitlement. */
	entitled: Set<string>;
	/** Bearers whose account has spent its quota window. */
	exhausted: Set<string>;
	stop(): void;
}

/**
 * Stand-in Anthropic endpoint, answering per credential.
 *
 * Rejects `speed` unless the request's bearer is in `entitled` — what the live
 * API does for an account without the fast-mode entitlement, so the provider's
 * one-shot fallback fires and a gateway that retained the lesson never asks
 * again. Rejects strict tool schemas with the grammar-too-large 400 whatever
 * the bearer: the compiled grammar is a property of the deployment, not the
 * account. Answers a bearer in `exhausted` with a usage-limit block, which is
 * what makes the gateway rotate mid-request.
 */
function startUpstream(): Upstream {
	const payloads: UpstreamPayload[] = [];
	const bearers: Array<string | undefined> = [];
	const entitled = new Set<string>();
	const exhausted = new Set<string>();
	const sse = `${SSE_EVENTS.map(event => `event: ${String(event.type)}\ndata: ${JSON.stringify(event)}`).join("\n\n")}\n\n`;
	const reject = (body: unknown, status: number): Response =>
		new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
	const server = Bun.serve({
		hostname: "127.0.0.1",
		port: 0,
		fetch: async (req): Promise<Response> => {
			// The usage-limit path also probes `/api/oauth/usage`; only inference
			// requests are the trace this fixture records.
			if (new URL(req.url).pathname !== "/v1/messages") return new Response("{}", { status: 404 });
			const payload = (await req.json()) as UpstreamPayload;
			// API-key credentials arrive as `x-api-key`, OAuth ones as a bearer.
			const bearer =
				req.headers.get("x-api-key") ?? req.headers.get("authorization")?.replace(/^Bearer /i, "") ?? undefined;
			payloads.push(payload);
			bearers.push(bearer);
			if (bearer !== undefined && exhausted.has(bearer)) return reject(USAGE_LIMIT_REJECTION, 429);
			if (payload.tools?.some(tool => tool.strict === true)) return reject(GRAMMAR_REJECTION, 400);
			if (payload.speed !== undefined && (bearer === undefined || !entitled.has(bearer))) {
				return reject(FAST_MODE_REJECTION, 400);
			}
			return new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream" } });
		},
	});
	return {
		url: `http://127.0.0.1:${server.port}`,
		payloads,
		bearers,
		entitled,
		exhausted,
		stop: () => {
			server.stop(true);
		},
	};
}

interface GatewayFixture {
	handle: AuthGatewayServerHandle;
	/** The gateway's credential source, so a test can switch the account under it. */
	storage: AuthStorage;
	stop(): Promise<void>;
	cleanup(): Promise<void>;
}

async function startGateway(
	model: Model<Api>,
	provider: string,
	options?: { sessionStateMax?: number; oauth?: boolean },
): Promise<GatewayFixture> {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "gw-session-state-"));
	const storage = await AuthStorage.create(path.join(dir, "auth.db"));
	// A runtime override outranks every stored row and blocks account pinning,
	// so the OAuth tests seed credentials instead.
	if (!options?.oauth) storage.setRuntimeApiKey(provider, "sk-ant-api-test");
	const handle = startAuthGateway({
		bind: "127.0.0.1:0",
		bearerTokens: ["test-token"],
		storage,
		resolveModel: () => model,
		version: "test",
		sessionStateMax: options?.sessionStateMax,
	});
	let stopped = false;
	const stop = async (): Promise<void> => {
		if (stopped) return;
		stopped = true;
		await handle.close();
	};
	return {
		handle,
		storage,
		stop,
		cleanup: async () => {
			await stop();
			storage.close();
			await fs.rm(dir, { recursive: true, force: true });
		},
	};
}

/** One store request for a client-keyed session, as a gateway handler builds it. */
function stateRequest(clientKey: string, account = "key:test-account"): AuthGatewaySessionStateRequest {
	return { clientKey, model: ANTHROPIC_MODEL, context: CONTEXT, account };
}

/**
 * One stored OAuth row. `orgId` is what lets two rows of the same account sit
 * side by side: `resolveCredentialIdentityKey` keys anthropic rows
 * `<base>|org:<id>`, so a Team seat and a personal plan of one login are two
 * credentials rather than one overwritten row.
 */
function oauthAccount(ids: { suffix: string; accountId: string; orgId: string }): OAuthCredential {
	return {
		type: "oauth",
		access: `access-${ids.suffix}`,
		refresh: `refresh-${ids.suffix}`,
		expires: Date.now() + 60 * 60_000,
		accountId: ids.accountId,
		email: "shared@example.com",
		orgId: ids.orgId,
	};
}

/**
 * Hand `getApiKey` the stored access token verbatim. Without this the OAuth
 * path would try to mint one against the real token endpoint.
 */
function mockOAuthAccess(): void {
	vi.spyOn(oauthUtils, "getOAuthApiKey").mockImplementation(async (provider, credentials) => {
		const credential = credentials[provider];
		return credential ? { newCredentials: credential, apiKey: credential.access } : null;
	});
}

/**
 * Switch a session onto a stored account, the way the account picker and the
 * usage-limit handoff both do. `orgId` picks between two rows of one account.
 */
function pinAccount(storage: AuthStorage, sessionId: string, accountId: string, orgId?: string): void {
	const target = storage
		.listOAuthAccounts("anthropic", sessionId)
		.find(account => account.accountId === accountId && (orgId === undefined || account.orgId === orgId));
	if (!target) throw new Error(`no stored account ${accountId}${orgId ? `/${orgId}` : ""}`);
	if (!storage.pinSessionOAuthAccount("anthropic", sessionId, target.credentialId)) {
		throw new Error(`could not pin account ${accountId}`);
	}
}

/**
 * One priority-tier turn through the pi-native route. Returns the status plus
 * the decoded envelope so a failed turn reports the upstream reason instead of
 * a bare number.
 */
async function priorityTurn(
	handle: AuthGatewayServerHandle,
	sessionId: string,
	modelId: string,
): Promise<{ status: number; body: unknown }> {
	const response = await fetch(`${handle.url}/v1/pi/stream`, {
		method: "POST",
		headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
		body: JSON.stringify({
			modelId,
			context: CONTEXT,
			options: { sessionId, serviceTier: "priority" },
			stream: false,
		}),
	});
	return { status: response.status, body: await response.json() };
}

/**
 * One priority-tier turn through a foreign-wire route, carrying the history the
 * caller supplies. No session key is sent, so the gateway has to work out which
 * conversation this is from the history itself.
 */
async function priorityChatHistory(
	handle: AuthGatewayServerHandle,
	modelId: string,
	messages: ReadonlyArray<{ role: "user" | "assistant"; content: string }>,
): Promise<{ status: number; body: unknown }> {
	const response = await fetch(`${handle.url}/v1/chat/completions`, {
		method: "POST",
		headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
		body: JSON.stringify({
			model: modelId,
			messages: [{ role: "system", content: "Stay concise." }, ...messages],
			service_tier: "priority",
			stream: false,
		}),
	});
	return { status: response.status, body: await response.json() };
}

/** A single-message turn: identical bodies land on the same logical session. */
async function priorityChatTurn(
	handle: AuthGatewayServerHandle,
	modelId: string,
	prompt: string,
): Promise<{ status: number; body: unknown }> {
	return priorityChatHistory(handle, modelId, [{ role: "user", content: prompt }]);
}

/**
 * One tool-carrying turn through the pi-native route. `bash` is on Anthropic's
 * strict-tool allowlist and the schema is a closed object, so the tool goes out
 * `strict: true` until the session learns the endpoint rejects the compiled
 * grammar.
 */
async function toolTurn(
	handle: AuthGatewayServerHandle,
	sessionId: string,
	modelId: string,
): Promise<{ status: number; body: unknown }> {
	const response = await fetch(`${handle.url}/v1/pi/stream`, {
		method: "POST",
		headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
		body: JSON.stringify({
			modelId,
			context: {
				...CONTEXT,
				tools: [
					{
						name: "bash",
						description: "Run a shell command",
						strict: true,
						parameters: {
							type: "object",
							properties: { command: { type: "string" } },
							required: ["command"],
							additionalProperties: false,
						},
					},
				],
			},
			options: { sessionId },
			stream: false,
		}),
	});
	return { status: response.status, body: await response.json() };
}

withOfficialAnthropicEndpoint();

describe("auth-gateway provider session state", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("carries a session's learned fast-mode fallback into its next request", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			// Turn one: asks for fast mode, gets rejected, retries without it.
			// Turn two: the lesson survived the request boundary, so it never asks
			// again — one wasted round-trip per session instead of one per turn.
			expect(upstream.payloads.map(payload => payload.speed)).toEqual(["fast", undefined, undefined]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("carries the learned fallback across requests on the foreign-wire routes", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			expect(await priorityChatTurn(gateway.handle, model.id, "Hi")).toMatchObject({ status: 200 });
			expect(await priorityChatTurn(gateway.handle, model.id, "Hi")).toMatchObject({ status: 200 });
			// Different conversation seed, so a different derived session: it asks
			// for fast mode on its own account.
			expect(await priorityChatTurn(gateway.handle, model.id, "Other")).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.speed)).toEqual([
				"fast",
				undefined,
				undefined,
				"fast",
				undefined,
			]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("keeps one session's fallback out of another session's requests", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });
			expect(await priorityTurn(gateway.handle, "session-b", model.id)).toMatchObject({ status: 200 });

			// Session B is a different conversation, possibly a different account:
			// it still asks for priority routing rather than inheriting A's
			// downgrade, then learns the same lesson on its own.
			expect(upstream.payloads.map(payload => payload.speed)).toEqual(["fast", undefined, "fast", undefined]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("gives two keyless conversations that share an opening their own retained state", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			// Conversation A opens and learns the fallback.
			expect(await priorityChatHistory(gateway.handle, model.id, [{ role: "user", content: "Hi" }])).toMatchObject({
				status: 200,
			});
			// A's second turn extends A's history, so it keeps A's lesson.
			expect(
				await priorityChatHistory(gateway.handle, model.id, [
					{ role: "user", content: "Hi" },
					{ role: "assistant", content: "ok" },
					{ role: "user", content: "and now about A" },
				]),
			).toMatchObject({ status: 200 });
			// Conversation B opened the same way but is a different chat. Sharing a
			// derived key with A would silence B's priority request off A's
			// rejection — and hand B whatever transport session A is on.
			expect(
				await priorityChatHistory(gateway.handle, model.id, [
					{ role: "user", content: "Hi" },
					{ role: "assistant", content: "ok" },
					{ role: "user", content: "and now about B" },
				]),
			).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.speed)).toEqual([
				"fast",
				undefined,
				undefined,
				"fast",
				undefined,
			]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("re-probes the account-scoped lesson after the session switches credentials", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			// What markUsageLimitReached does to a session: same conversation, next
			// credential. Fast mode is an entitlement of the account that was
			// rejected, not of the endpoint, so the new one has to be asked.
			gateway.storage.setRuntimeApiKey("anthropic", "sk-ant-api-sibling");
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });
			// Still the sibling: a switch re-probes once, it does not re-probe every
			// turn afterwards.
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.speed)).toEqual([
				"fast",
				undefined,
				undefined,
				"fast",
				undefined,
				undefined,
			]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("keeps the endpoint-scoped lesson a session learned when its credential switches", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic");
		try {
			// Turn one pays the grammar-too-large rejection and retries non-strict.
			expect(await toolTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			// Same conversation, next credential. The compiled tool grammar is a
			// property of the deployment, so keying the retained state by credential
			// — or rebuilding it wholesale on a switch — makes this turn re-pay a
			// round trip the session already paid for.
			gateway.storage.setRuntimeApiKey("anthropic", "sk-ant-api-sibling");
			expect(await toolTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.tools?.some(tool => tool.strict === true))).toEqual([
				true,
				false,
				false,
			]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("re-probes the account-scoped lesson on the retry that rotated, not a turn later", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic", { oauth: true });
		try {
			mockOAuthAccess();
			await gateway.storage.set("anthropic", [
				oauthAccount({ suffix: "alpha", accountId: "acc-alpha", orgId: "org-alpha" }),
				oauthAccount({ suffix: "beta", accountId: "acc-beta", orgId: "org-beta" }),
			]);
			// The sibling's plan includes fast mode; the first account's does not.
			upstream.entitled.add("access-beta");
			pinAccount(gateway.storage, "session-a", "acc-alpha");

			// Turn one learns that this account has no fast-mode entitlement.
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			// Turn two: the account's quota window is spent, so the gateway blocks
			// it and the in-request retry rotates to the sibling. Comparing accounts
			// only at the next request boundary leaves that retry running on the
			// blocked account's verdict, and the rotated turn silently gives up the
			// priority routing the client asked for and the new account can serve.
			upstream.exhausted.add("access-alpha");
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.speed)).toEqual(["fast", undefined, undefined, "fast"]);
			expect(upstream.bearers).toEqual(["access-alpha", "access-alpha", "access-alpha", "access-beta"]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("treats two org-scoped subscriptions of one account as different accounts", async () => {
		const upstream = startUpstream();
		const model = makeAnthropicModel(upstream.url);
		const gateway = await startGateway(model, "anthropic", { oauth: true });
		try {
			mockOAuthAccess();
			// One Anthropic login holding two org-scoped subscriptions: same
			// account id and email, one Team seat and one personal plan, stored
			// side by side because `resolveCredentialIdentityKey` keys anthropic
			// rows `<base>|org:<id>`.
			await gateway.storage.set("anthropic", [
				oauthAccount({ suffix: "team", accountId: "acc-shared", orgId: "org-team" }),
				oauthAccount({ suffix: "personal", accountId: "acc-shared", orgId: "org-personal" }),
			]);
			// The personal plan includes fast mode; the Team seat does not.
			upstream.entitled.add("access-personal");
			pinAccount(gateway.storage, "session-a", "acc-shared", "org-team");
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			// Same account id and email, different subscription. An identity that
			// stops at the shared base reads this as no switch at all, and the turn
			// runs on the other subscription's entitlement verdict.
			pinAccount(gateway.storage, "session-a", "acc-shared", "org-personal");
			expect(await priorityTurn(gateway.handle, "session-a", model.id)).toMatchObject({ status: 200 });

			expect(upstream.payloads.map(payload => payload.speed)).toEqual(["fast", undefined, "fast"]);
			expect(upstream.bearers).toEqual(["access-team", "access-team", "access-personal"]);
		} finally {
			await gateway.cleanup();
			upstream.stop();
		}
	});

	it("closes the provider state it evicts at the session ceiling", () => {
		const store = new AuthGatewaySessionStateStore(1);
		const closed: string[] = [];
		const first = store.acquire(stateRequest("session-a"));
		first.states.set("probe", { close: () => closed.push("session-a") });
		first.release();

		const again = store.acquire(stateRequest("session-a"));
		expect(again.states).toBe(first.states);
		expect(closed).toEqual([]);
		again.release();

		store.acquire(stateRequest("session-b")).release();

		// Dropping an entry without closing it leaks the sockets and timers the
		// ceiling exists to cap, and handing the dropped map back would resurrect
		// state whose `close()` already ran.
		expect(closed).toEqual(["session-a"]);
		expect(store.size).toBe(1);
		expect(store.acquire(stateRequest("session-a")).states).not.toBe(first.states);
	});

	it("leaves an entry a request is still holding alone, and closes it on release", () => {
		const store = new AuthGatewaySessionStateStore(1);
		const closed: string[] = [];
		const live = store.acquire(stateRequest("session-live"));
		live.states.set("probe", { close: () => closed.push("live") });

		// A second conversation arrives while the first one's stream is still
		// running. Making room by closing that entry would tear down the flags and
		// sockets the stream is mid-turn on, so the ceiling gives instead.
		const next = store.acquire(stateRequest("session-next"));
		next.states.set("probe", { close: () => closed.push("next") });
		expect(closed).toEqual([]);
		expect(store.size).toBe(2);

		// The stream ends. Its entry is the least recently acquired, so the
		// deferred eviction takes it and the ceiling is back in force.
		live.release();
		expect(closed).toEqual(["live"]);
		expect(store.size).toBe(1);

		next.release();
		expect(closed).toEqual(["live"]);
		expect(store.size).toBe(1);
	});

	it("hands the retained state back when a request throws, so the ceiling still applies", async () => {
		registerMockApi();
		const mock = createMockModel({ provider: "openrouter", id: "gw-session-throw" });
		const gateway = await startGateway(mock, "openrouter", { sessionStateMax: 1 });
		try {
			mock.push(() => {
				throw new Error("upstream transport exploded");
			});
			const failed = await fetch(`${gateway.handle.url}/v1/pi/stream`, {
				method: "POST",
				headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
				body: JSON.stringify({
					modelId: mock.id,
					context: CONTEXT,
					options: { sessionId: "throwing-session" },
					stream: false,
				}),
			});
			expect(failed.status).toBeGreaterThanOrEqual(400);
			await failed.json();

			// The map the failed request was handed IS its retained entry.
			const states = mock.calls[0]?.options?.providerSessionState;
			expect(states).toBeDefined();
			const closed: string[] = [];
			states?.set("probe", { close: () => closed.push("probe") });

			// At a ceiling of one, a second session can only be admitted if the
			// thrown request gave its entry back. A claim leaked on the error path
			// pins that entry for the life of the process.
			mock.push({ content: ["ok"] });
			const next = await fetch(`${gateway.handle.url}/v1/pi/stream`, {
				method: "POST",
				headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
				body: JSON.stringify({
					modelId: mock.id,
					context: CONTEXT,
					options: { sessionId: "next-session" },
					stream: false,
				}),
			});
			expect(next.status).toBe(200);
			await next.json();

			expect(closed).toEqual(["probe"]);
		} finally {
			await gateway.cleanup();
			clearCustomApis();
		}
	});

	it("closes every retained provider state when the gateway shuts down", async () => {
		registerMockApi();
		const mock = createMockModel({ provider: "openrouter", id: "gw-session-drain" });
		const gateway = await startGateway(mock, "openrouter");
		try {
			mock.push({ content: ["ok"] });
			const response = await fetch(`${gateway.handle.url}/v1/pi/stream`, {
				method: "POST",
				headers: { Authorization: "Bearer test-token", "Content-Type": "application/json" },
				body: JSON.stringify({
					modelId: mock.id,
					context: CONTEXT,
					options: { sessionId: "drain-session" },
					stream: false,
				}),
			});
			expect(response.status).toBe(200);
			await response.json();

			// The map the provider was handed IS the gateway's retained entry.
			const states = mock.calls[0]?.options?.providerSessionState;
			expect(states).toBeDefined();
			const closed: string[] = [];
			states?.set("probe", { close: () => closed.push("probe") });

			await gateway.stop();

			expect(closed).toEqual(["probe"]);
		} finally {
			await gateway.cleanup();
			clearCustomApis();
		}
	});
});
