
import container_manager_mcp


def test_mcp_available_flag_resolves_without_raising():
    assert isinstance(container_manager_mcp._MCP_AVAILABLE, bool)




def test_known_mcp_server_symbol_is_reachable():
    assert hasattr(container_manager_mcp, "ctx_log")
    assert callable(container_manager_mcp.ctx_log)


def test_known_core_symbol_is_reachable():
    assert hasattr(container_manager_mcp, "create_manager")
    assert callable(container_manager_mcp.create_manager)


def test_unknown_attribute_raises_with_exact_message():
    try:
        container_manager_mcp.totally_bogus_symbol_xyz
    except AttributeError as e:
        assert (
            str(e)
            == "module 'container_manager_mcp' has no attribute 'totally_bogus_symbol_xyz'"
        )
    else:
        raise AssertionError("expected AttributeError")
