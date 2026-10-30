# CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION — Transport through the SDK

Owner: container-manager-mcp. Cross-repo IDs: `SCION-CENTRALIZED-CAPABILITIES-R001`, `SCION-CENTRALIZED-CAPABILITIES-R008`, `SCION-CENTRALIZED-CAPABILITIES-R012`, `SCION-CENTRALIZED-CAPABILITIES-R040`.

Extends: [SCION-CENTRALIZED-CAPABILITIES](https://github.com/Knuckles-Team/scion-sdk/blob/main/specs/centralized-capabilities/spec.md)

## Purpose and user stories

The connector must reach the cluster identity provider through the SDK. The SDK owns the governed HTTP client, retry and TLS profiles. Docker, Podman and Kubernetes calls use their native clients and are out of scope.

An operator imports a kube context with an OIDC login. Transport security comes from one TLS profile, not from a connector flag.

## Requirements and acceptance

The rows are in [requirements.md](requirements.md). The proving tests are in [test-spec.md](test-spec.md).

## Why a new spec

The search covered `specs/` of this repository and the SDK centralized-capabilities spec. This repository had no spec for transport. The SDK owns the capability. This spec states how this connector uses it.

## Scope and interfaces

In scope: the connector's HTTP client, TLS handling and dependency floor.

Out of scope: the tool surface, which stays one tool per API operation, and the SDK modules themselves.
