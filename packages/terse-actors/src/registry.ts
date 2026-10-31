import { Actor, Atomic, Persisted } from "durable-actors";
import {
  artifactSchema,
  id,
  type Document,
  type RegistryState,
  type SignedRequest,
} from "./contracts.js";
import { authorize, commit, document, duplicate, events, fail, journal } from "./core.js";

const actorName = "ArtifactRegistryActor";
export class ArtifactRegistryActor extends Actor {
  @Persisted private state: RegistryState = { ...journal(), artifacts: {} };

  @Atomic
  async register(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "register",
      ["worker", "evaluator", "data"],
      artifactSchema,
    );
    if (request.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Registry belongs to another campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const allowed = {
      worker: ["checkpoint", "trajectory", "report"],
      evaluator: ["evaluation", "report"],
      data: ["dataset"],
    };
    if (!(allowed[auth.role as keyof typeof allowed] ?? []).includes(auth.payload.kind))
      fail("ARTIFACT_ROLE", "This identity cannot register that artifact kind");
    if (Object.keys(this.state.artifacts).length >= 4000)
      fail("ARTIFACT_LIMIT", "Registry capacity reached");
    if (this.state.artifacts[auth.payload.artifactId])
      fail("ARTIFACT_IMMUTABLE", "Artifact identifiers cannot be overwritten");
    const artifact = { ...auth.payload, registeredBy: auth.keyId };
    const result = document(artifact);
    this.state = commit(
      this.db,
      { ...this.state, artifacts: { ...this.state.artifacts, [artifact.artifactId]: artifact } },
      auth,
      "ARTIFACT_REGISTERED",
      result,
      {
        artifactId: artifact.artifactId,
        experimentId: artifact.experimentId,
        sha256: artifact.sha256,
        kind: artifact.kind,
      },
    );
    return result;
  }
  async getArtifact(artifactId: string): Promise<Document | null> {
    id.parse(artifactId);
    const artifact = this.state.artifacts[artifactId];
    return artifact ? document(artifact) : null;
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
