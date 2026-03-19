#!/usr/bin/env python3
"""
Kimi Session Manager - Automatic Session State Tracking
Saves session context, progress, and status automatically
"""

import json
import os
from datetime import datetime
from pathlib import Path


class SessionManager:
    """Manages automatic session state saving"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.home() / "retin-verify"
        self.state_file = self.project_root / ".kimi" / "session_state.json"
        self.status_file = self.project_root / "CURRENT_STATUS.txt"
        self.history_file = self.project_root / ".kimi" / "session_history.jsonl"
        
        # Ensure .kimi directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, task_name, status, progress, next_steps, notes=""):
        """
        Save current session state
        
        Args:
            task_name: Name of current task
            status: Current status (in_progress, complete, blocked)
            progress: Progress description
            next_steps: What to do next
            notes: Additional notes
        """
        timestamp = datetime.now().isoformat()
        
        # Create state object
        state = {
            "last_session": {
                "timestamp": timestamp,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task": task_name,
                "status": status,
                "progress": progress,
                "next_steps": next_steps,
                "notes": notes
            },
            "session_count": self._get_session_count() + 1
        }
        
        # Save to JSON state file
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Append to history
        with open(self.history_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "task": task_name,
                "status": status,
                "progress": progress
            }) + "\n")
        
        # Update human-readable status file
        self._update_status_file(state["last_session"])
        
        return state
    
    def load_session(self):
        """Load last session state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_summary(self):
        """Get quick summary of last session"""
        state = self.load_session()
        if state and "last_session" in state:
            s = state["last_session"]
            return f"""
╔══════════════════════════════════════════════════════════════╗
║              PREVIOUS SESSION SUMMARY                          ║
╠══════════════════════════════════════════════════════════════╣
║ Date: {s['date']}
║ Task: {s['task']}
║ Status: {s['status']}
╠══════════════════════════════════════════════════════════════╣
║ Progress:
║   {s['progress'][:50]}...
╠══════════════════════════════════════════════════════════════╣
║ Next Steps:
║   {s['next_steps'][:50]}...
╚══════════════════════════════════════════════════════════════╝
"""
        return "No previous session found."
    
    def _get_session_count(self):
        """Get total session count from history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return sum(1 for _ in f)
        return 0
    
    def _update_status_file(self, session_data):
        """Update human-readable CURRENT_STATUS.txt"""
        content = f"""# Current Project Status
# Auto-generated: {session_data['date']}

## Last Session
- **Date:** {session_data['date']}
- **Task:** {session_data['task']}
- **Status:** {session_data['status']}

## Progress
{session_data['progress']}

## Next Steps
{session_data['next_steps']}

## Notes
{session_data.get('notes', 'None')}

---
# To update this file, use:
# python3 .kimi/session_manager.py save "Task Name" "status" "progress" "next_steps"
"""
        with open(self.status_file, 'w') as f:
            f.write(content)


def auto_save_on_exit():
    """
    Auto-save hook - call this before session ends
    This should be called automatically when Kimi session ends
    """
    manager = SessionManager()
    
    # Try to detect what was being worked on
    git_status = os.popen("git status --short 2>/dev/null").read().strip()
    recent_commits = os.popen("git log --oneline -3 2>/dev/null").read().strip()
    
    # Create auto-save entry
    manager.save_session(
        task_name="Auto-saved session",
        status="interrupted",
        progress=f"Git status:\n{git_status[:200]}",
        next_steps="Review git status and continue work",
        notes=f"Recent commits:\n{recent_commits}"
    )
    
    print("✅ Session auto-saved")


def interactive_save():
    """Interactive session save"""
    import sys
    
    manager = SessionManager()
    
    if len(sys.argv) >= 5:
        # Command line mode
        task = sys.argv[2]
        status = sys.argv[3]
        progress = sys.argv[4]
        next_steps = sys.argv[5] if len(sys.argv) > 5 else "Continue from where left off"
        notes = sys.argv[6] if len(sys.argv) > 6 else ""
    else:
        # Interactive mode
        print("═" * 60)
        print("  SAVE SESSION")
        print("═" * 60)
        task = input("Task name: ")
        status = input("Status (in_progress/complete/blocked): ") or "in_progress"
        progress = input("Progress made: ")
        next_steps = input("Next steps: ")
        notes = input("Notes (optional): ")
    
    manager.save_session(task, status, progress, next_steps, notes)
    print("\n✅ Session saved successfully!")
    print(f"📄 Updated: {manager.status_file}")
    print(f"📄 Updated: {manager.state_file}")


def show_summary():
    """Show last session summary"""
    manager = SessionManager()
    print(manager.get_summary())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 session_manager.py save 'Task Name' 'status' 'progress' 'next_steps' [notes]")
        print("  python3 session_manager.py show")
        print("  python3 session_manager.py auto-save")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "save":
        interactive_save()
    elif command == "show":
        show_summary()
    elif command == "auto-save":
        auto_save_on_exit()
    else:
        print(f"Unknown command: {command}")
