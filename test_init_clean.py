#!/usr/bin/env python3
"""Test /init --erase command with clean database state."""

import os
import sys
import shutil

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_clean_init():
    """Test initializing a database from scratch."""
    # Clean up any existing database
    episodic_dir = os.path.expanduser("~/.episodic")
    if os.path.exists(episodic_dir):
        print(f"Removing existing directory: {episodic_dir}")
        shutil.rmtree(episodic_dir)
    
    # Now import and initialize
    from episodic.db import initialize_db
    from episodic.conversation import ConversationManager
    
    print("\nInitializing fresh database...")
    try:
        # Don't use erase=True when there's no database
        initialize_db(erase=False)
        print("✅ Database initialized successfully")
        
        # Try to create a conversation manager
        print("\nCreating ConversationManager...")
        cm = ConversationManager()
        cm.initialize_conversation()
        print("✅ ConversationManager created successfully")
        
        # Test adding a message
        print("\nTesting message addition...")
        from episodic.db import insert_node
        node_id, short_id = insert_node("Test message", None, role="user")
        print(f"✅ Added test node: {short_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_init()
    sys.exit(0 if success else 1)