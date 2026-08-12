use super::support::*;
use super::*;
use crate::tools::todo::{TodoItem, TodoPriority, TodoState, TodoStatus};
use xai_grok_tools::implementations::grok_build::todo::SubagentTodoBindings;
use xai_grok_tools::types::resources::State;

fn todo_state() -> TodoState {
    let mut state = TodoState::default();
    state.push(
        "work".to_string(),
        TodoItem {
            content: "Do work".to_string(),
            priority: TodoPriority::Medium,
            status: TodoStatus::Pending,
            meta: None,
        },
    );
    state
}

fn next_plan_status(
    events: &mut tokio::sync::mpsc::UnboundedReceiver<SessionEvent>,
) -> acp::PlanEntryStatus {
    loop {
        let SessionEvent::Notification(SessionNotification::Acp(notification)) =
            events.try_recv().expect("expected queued Plan update")
        else {
            continue;
        };
        if let acp::SessionUpdate::Plan(plan) = notification.update {
            return plan.entries[0].status.clone();
        }
    }
}

#[tokio::test]
async fn confirmed_start_and_success_update_state_and_standard_plan() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let state_dir = tempfile::tempdir().unwrap();
            let (gateway_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (actor, mut events) =
                create_test_actor_ex(0, 256_000, 85, gateway_tx, persistence_tx).await;
            *actor.agent.borrow_mut() =
                test_agent_default_with_state_path(state_dir.path().join("tool_state.json")).await;
            let bridge = actor.tool_bridge_handle();
            bridge.update_resource(State(todo_state())).await;
            bridge
                .update_resource(State(SubagentTodoBindings::default()))
                .await;

            assert_eq!(
                actor.mark_subagent_todo_started("work", "run-1").await,
                Some(1)
            );
            assert_eq!(
                bridge
                    .read_resource::<State<TodoState>>()
                    .await
                    .unwrap()
                    .0
                    .status("work"),
                Some(TodoStatus::InProgress)
            );
            assert_eq!(
                next_plan_status(&mut events),
                acp::PlanEntryStatus::InProgress
            );

            actor.finish_subagent_todo("run-1", true).await;
            assert_eq!(
                bridge
                    .read_resource::<State<TodoState>>()
                    .await
                    .unwrap()
                    .0
                    .status("work"),
                Some(TodoStatus::Completed)
            );
            assert_eq!(
                next_plan_status(&mut events),
                acp::PlanEntryStatus::Completed
            );
        })
        .await;
}

#[tokio::test]
async fn failure_releases_current_owner_and_stale_success_is_ignored() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let state_dir = tempfile::tempdir().unwrap();
            let (gateway_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (actor, mut events) =
                create_test_actor_ex(0, 256_000, 85, gateway_tx, persistence_tx).await;
            *actor.agent.borrow_mut() =
                test_agent_default_with_state_path(state_dir.path().join("tool_state.json")).await;
            let bridge = actor.tool_bridge_handle();
            bridge.update_resource(State(todo_state())).await;
            bridge
                .update_resource(State(SubagentTodoBindings::default()))
                .await;

            assert_eq!(
                actor.mark_subagent_todo_started("work", "old").await,
                Some(1)
            );
            assert_eq!(
                actor.mark_subagent_todo_started("work", "retry").await,
                Some(2)
            );
            assert_eq!(
                actor.mark_subagent_todo_started("missing", "bad").await,
                None
            );
            actor.finish_subagent_todo("old", true).await;
            actor.finish_subagent_todo("retry", false).await;
            actor.finish_subagent_todo("old", true).await;

            assert_eq!(
                bridge
                    .read_resource::<State<TodoState>>()
                    .await
                    .unwrap()
                    .0
                    .status("work"),
                Some(TodoStatus::Pending)
            );
            let statuses = std::iter::from_fn(|| events.try_recv().ok())
                .filter_map(|event| match event {
                    SessionEvent::Notification(SessionNotification::Acp(notification)) => {
                        match notification.update {
                            acp::SessionUpdate::Plan(plan) => Some(plan.entries[0].status.clone()),
                            _ => None,
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                statuses,
                vec![
                    acp::PlanEntryStatus::InProgress,
                    acp::PlanEntryStatus::InProgress,
                    acp::PlanEntryStatus::Pending,
                ]
            );
        })
        .await;
}

#[tokio::test]
async fn failed_start_persistence_rolls_back_without_publishing_a_plan() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let state_dir = tempfile::tempdir().unwrap();
            let missing_parent = state_dir.path().join("missing");
            let (gateway_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (actor, mut events) =
                create_test_actor_ex(0, 256_000, 85, gateway_tx, persistence_tx).await;
            *actor.agent.borrow_mut() =
                test_agent_default_with_state_path(missing_parent.join("tool_state.json")).await;
            let bridge = actor.tool_bridge_handle();
            bridge.update_resource(State(todo_state())).await;
            bridge
                .update_resource(State(SubagentTodoBindings::default()))
                .await;

            assert_eq!(
                actor.mark_subagent_todo_started("work", "run-1").await,
                None
            );
            assert_eq!(
                bridge
                    .read_resource::<State<TodoState>>()
                    .await
                    .unwrap()
                    .0
                    .status("work"),
                Some(TodoStatus::Pending)
            );
            assert!(
                !bridge
                    .read_resource::<State<SubagentTodoBindings>>()
                    .await
                    .unwrap()
                    .0
                    .owns_subagent("run-1")
            );
            assert!(events.try_recv().is_err());
        })
        .await;
}

#[tokio::test]
async fn terminal_persistence_failure_still_publishes_the_completed_plan() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let state_dir = tempfile::tempdir().unwrap();
            let state_path = state_dir.path().join("tool_state.json");
            let (gateway_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, _) = tokio::sync::mpsc::unbounded_channel();
            let (actor, mut events) =
                create_test_actor_ex(0, 256_000, 85, gateway_tx, persistence_tx).await;
            *actor.agent.borrow_mut() = test_agent_default_with_state_path(state_path).await;
            let bridge = actor.tool_bridge_handle();
            bridge.update_resource(State(todo_state())).await;
            bridge
                .update_resource(State(SubagentTodoBindings::default()))
                .await;

            assert_eq!(
                actor.mark_subagent_todo_started("work", "run-1").await,
                Some(1)
            );
            assert_eq!(
                next_plan_status(&mut events),
                acp::PlanEntryStatus::InProgress
            );
            std::fs::remove_dir_all(state_dir.path()).unwrap();

            actor.finish_subagent_todo("run-1", true).await;
            assert_eq!(
                bridge
                    .read_resource::<State<TodoState>>()
                    .await
                    .unwrap()
                    .0
                    .status("work"),
                Some(TodoStatus::Completed)
            );
            assert_eq!(
                next_plan_status(&mut events),
                acp::PlanEntryStatus::Completed
            );
        })
        .await;
}
