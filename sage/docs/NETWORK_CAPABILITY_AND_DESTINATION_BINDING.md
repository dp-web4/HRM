# Network Capability and Destination Binding

**Status:** conceptual security architecture / research direction  
**Companion:** [`BEING_STACK_VISION.md`](BEING_STACK_VISION.md)  
**Motivating disclosure:** [CVE-2026-85666](https://www.cve.org/CVERecord?id=CVE-2026-85666), OGX <= 1.3.1 MCP `server_url` SSRF

SAGE's A2-class harness direction needs one invariant stated more explicitly:

> **The destination is part of the act.**

An authorization such as "use MCP", "call this tool", or even "invoke service X" is incomplete if caller-controlled data can still choose where the privileged runtime actually connects.

This is a general agent-runtime problem, not an MCP-specific one.

## 1. Why this matters

CVE-2026-85666 affected OGX (formerly Llama Stack) through version 1.3.1. Its OpenAI-compatible `/v1/responses` path accepted an MCP tool definition containing a caller-controlled `server_url`; the server then connected to that destination without applying the private-address validation used on sibling URL paths. In the default unauthenticated starter configuration, a remote caller could induce the runtime to connect from its own network position to loopback, private, link-local or cloud-metadata destinations and could supply headers/authorization material for the request.

The direct vulnerability is SSRF. The architectural lesson is broader:

- the model/caller names an apparently ordinary tool parameter;
- the runtime interprets that parameter as **network authority**;
- the actual privileged effect occurs below the layer where the request may have been classified or approved.

This is the network-side sibling of inference-template poisoning: the dangerous semantic boundary sits below what cognition perceives as the action.

## 2. SAGE invariant

For any harness-mediated network effect, authorization should bind the **resolved execution target**, not merely the tool name or caller prose.

Conceptually:

```text
cognition proposes
  "use capability C with input I"
          |
          v
SAGE harness resolves
  capability identity
  destination / audience
  credential authority
  redirect / resolution policy
          |
          v
authorize exact resolved act
          |
          v
executor performs only that act
          |
          v
record actual destination + outcome
```

A post-authorization change to the destination is a different act and requires a new decision.

## 3. What should be bound

The exact schema can evolve, but a consequential outbound network act should eventually distinguish at least:

- logical capability/tool identity;
- requested destination descriptor;
- canonical destination identity (service/audience where known);
- scheme and transport;
- resolved host/address class at execution time;
- credential/token audience and scope;
- redirect policy and observed redirect chain;
- calling being / role / delegation;
- inference/substrate epoch that proposed the act;
- harness decision/evidence reference;
- observed execution result.

The point is not to freeze DNS forever. The point is to make authority explicit across resolution. A hostname may legitimately map to changing addresses; a decision should define what changes remain within the authorized audience and what changes constitute a new target.

## 4. Credentials are destination capabilities

A credential is not merely a secret string. Its useful security meaning includes **where it may be presented**.

The SAGE vault should therefore prefer operations shaped like:

```text
broker_request(
    capability = service-X,
    audience = api.service-x.example,
    operation = read_resource,
    limits = ...
)
```

rather than:

```text
get_secret("service-X-token")
```

Where raw export remains necessary, it should be exceptional and visibly lower-assurance.

A caller-controlled URL must never silently rebind a credential intended for one audience to another destination.

## 5. Network resolution is part of the trust boundary

A robust implementation eventually needs to reason about more than the literal URL string:

- loopback/private/link-local/metadata ranges;
- DNS rebinding or resolution changes between check and use;
- redirects to a different authority;
- proxy behavior;
- IPv4/IPv6 equivalence and alternate address encodings;
- userinfo or header-based credential forwarding;
- local Unix-socket / named-pipe equivalents;
- service discovery that resolves a logical name to a concrete endpoint.

These are executor/harness concerns. Cognition should not be expected to defend itself by remembering URL hygiene rules.

## 6. Relationship to Hestia

SAGE should not invent a second external authority system.

For boundary-crossing acts, the SAGE harness should produce a resolved action description that Hestia can bind into its existing identity, role, delegation, law and evidence model. Hestia's decision should cover the actual audience/target sufficiently tightly that an executor cannot take an approval for one destination and spend it on another.

The separation remains:

- **SAGE** resolves and mediates the being's internal capability request;
- **Hestia** provides canonical external authorization/evidence;
- **the executor/relying service** enforces the bound decision at the point of effect.

That is the A2 shape.

## 7. Relationship to Hub

A Hub may eventually help beings discover services, tools, MCP servers, agents or other capability-bearing endpoints. Discovery must not become execution authority.

A Hub-supplied endpoint should be treated as a **provenance-bearing descriptor/claim**:

- who advertised it;
- under which identity/role;
- which service/capability it claims to represent;
- what evidence or reputation accompanies the claim.

Accepting, discovering or receiving that descriptor does **not** authorize SAGE to connect to it. The connection still crosses the SAGE harness and, where externally consequential, Hestia law.

## 8. Minimum red-team cases

The eventual A2 harness test suite should include:

1. caller supplies an MCP/tool URL pointing to loopback;
2. caller supplies private or link-local/cloud-metadata destination;
3. approved public hostname redirects to private address;
4. DNS answer changes between authorization and connection;
5. approved capability attempts to forward a credential to a different audience;
6. tool name and schema stay constant while `server_url` changes after decision;
7. Hub message advertises a validly signed but malicious endpoint;
8. compromised cognitive worker attempts direct socket access around the broker.

The expected result is not necessarily "deny every private destination." Local services are legitimate. The expected result is that **the authority to reach that destination is explicit, scoped, attributable and enforced outside cognition.**

## 9. What this does not imply

This does not require a custom kernel or inference runtime now. Most of the useful control can first be built with a brokered network path, authenticated local IPC, OS egress restrictions, typed destination metadata, vault audience binding and Hestia decision binding.

A specialized runtime or OS becomes justified only if concrete invariants remain impossible or unacceptably fragile on the commodity substrate.
