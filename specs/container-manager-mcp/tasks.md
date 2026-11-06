# CONN-CONTAINER-MANAGER-MCP — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

Each task line reads `- [ ] T001 [P] [US1] <ID>-R012.4 description`. Order is producer first.

- [x] T001 [US1] CONN-CONTAINER-MANAGER-MCP-R001 Per-call context.
- [x] T002 [US1] CONN-CONTAINER-MANAGER-MCP-R001.1 Advertised argument.
- [x] T003 [US1] CONN-CONTAINER-MANAGER-MCP-R001.2 Named context selects the manager.
- [x] T004 [US1] CONN-CONTAINER-MANAGER-MCP-R001.3 Default manager.
- [x] T005 [US1] CONN-CONTAINER-MANAGER-MCP-R001.4 Wiring.
- [x] T006 [US1] CONN-CONTAINER-MANAGER-MCP-R001.5 Hot-path proof.
- [x] T007 [US1] CONN-CONTAINER-MANAGER-MCP-R002 Deployed workload access contract.
- [x] T008 [US1] CONN-CONTAINER-MANAGER-MCP-R003 Inventory privacy.
- [x] T009 [US1] CONN-CONTAINER-MANAGER-MCP-R004 Shared inventory path.
- [ ] T010 [US1] CONN-CONTAINER-MANAGER-MCP-R005 Workload recorded response.
- [ ] T011 [US1] CONN-CONTAINER-MANAGER-MCP-R008 k8s_list_services matches the declared Deployment contract.
- [x] T012 [US1] CONN-CONTAINER-MANAGER-MCP-R006 Windows portability.
- [x] T013 [US1] CONN-CONTAINER-MANAGER-MCP-R007 Destructive confirmation fails closed.
- [x] T014 [US1] CONN-CONTAINER-MANAGER-MCP-R009 Cluster-wide reads.
- [x] T015 [US1] CONN-CONTAINER-MANAGER-MCP-R010 Secret values stay hidden by default.
- [x] T016 [US1] CONN-CONTAINER-MANAGER-MCP-R011 Rollouts behave like kubectl.
- [x] T017 [US1] CONN-CONTAINER-MANAGER-MCP-R012 ExternalSecret force-sync proves the refresh.
- [x] T018 [US1] CONN-CONTAINER-MANAGER-MCP-R013 GitOps tie-in.
- [x] T019 [US1] CONN-CONTAINER-MANAGER-MCP-R014 Any-kind apply, read and delete.
- [x] T020 [US1] CONN-CONTAINER-MANAGER-MCP-R015 Served content.
- [x] T021 [US1] CONN-CONTAINER-MANAGER-MCP-R016 Ontology contracts and core roll-up.
- [x] T022 [US1] CONN-CONTAINER-MANAGER-MCP-R017 Deployments and logs in any namespace.
