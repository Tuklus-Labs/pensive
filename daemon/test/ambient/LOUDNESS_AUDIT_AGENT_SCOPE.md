# Loudness audit: first-class reader

Sweep of new assertions. Four-box: names the rule, enough state to debug,
grep-unique, present-tense.

| Test | Rule named in message | Exemption |
|------|------------------------|-----------|
| test_agent_active_excludes_other_agents_recent_work | agent-scope invariant | |
| test_agent_active_survives_foreign_recent_tail | foreign-tail window invariant | |
| test_blank_agent_is_unscoped_fleet_active | unscoped-fleet / blank-agent contract | |
| test_pins_remain_shared_across_agents | shared-pin invariant | |
| test_document_chunk_never_in_active | chunk-in-active invariant | |
| test_brief_with_no_matching_sections_is_minimal | named-agent-empty-active rule | |
| test_grok_hook_daemon_down_preserves_existing_brief | grok-hook fail-open / preserve-on-failure | |
| test_grok_hook_writes_brief_via_fake_curl | grok-hook write / stdout-ignored | |
| test_filter_agent_keeps_stamped_drops_null_and_other | agent-filter / unscoped-recall | |
| test_pensive_recall_passes_l2_tier | pensive_recall L2-default | |
| test_handle_recall_forwards_kinds_instead_of_default_tier | handle_recall kinds-honor | |
| test_handle_recall_does_not_forward_transport_agent | transport-autoscope | |
| test_recall_records_forwards_agent_only_when_set | unset-agent-omit / agent-forwarding | |

Existing `assert out == EMPTY_BRIEF` in test_empty_store predates this slice.
Not rewritten.

No exemption uses "it is simple."
