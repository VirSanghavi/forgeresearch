import { randomUUID, sign, type KeyObject } from "node:crypto";
import { createActorTransport, type DurableActorsClientOptions } from "durable-actors/backend";
import { type ActorName, type Document, type SignedRequest } from "./contracts.js";
import { canonical } from "./core.js";

export interface SigningIdentity {
  keyId: string;
  privateKey: KeyObject;
}
/** Backend-only: private signing keys never enter actor state, requests, or browser clients. */
export class ForgeActorClient {
  private readonly transport;
  constructor(
    private readonly connection: DurableActorsClientOptions & { projectId: string },
    private readonly identity: SigningIdentity,
  ) {
    this.transport = createActorTransport(connection);
  }
  signed(
    actorName: ActorName,
    actorId: string,
    campaignId: string,
    method: string,
    payload: Document,
    requestId: string = randomUUID(),
    validityMs = 60_000,
  ): SignedRequest {
    const now = Date.now();
    const unsigned = {
      version: 1 as const,
      projectId: this.connection.projectId,
      actorName,
      actorId,
      campaignId,
      method,
      payload,
      requestId,
      keyId: this.identity.keyId,
      issuedAtMs: now,
      expiresAtMs: now + validityMs,
    };
    return {
      ...unsigned,
      signature: sign(null, Buffer.from(canonical(unsigned)), this.identity.privateKey).toString(
        "base64url",
      ),
    };
  }
  async send(request: SignedRequest): Promise<Document> {
    return (await this.transport.invoke(request.actorName, request.actorId, request.method, [
      request,
    ])) as Document;
  }
  async command(
    actorName: ActorName,
    actorId: string,
    campaignId: string,
    method: string,
    payload: Document,
    requestId?: string,
  ): Promise<Document> {
    return this.send(this.signed(actorName, actorId, campaignId, method, payload, requestId));
  }
  async read(
    actorName: ActorName,
    actorId: string,
    method: string,
    args: unknown[] = [],
  ): Promise<Document | null> {
    return (await this.transport.invoke(actorName, actorId, method, args)) as Document | null;
  }
}
