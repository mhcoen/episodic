# Window Detection Score Storage Issue Summary

## Current State
1. The `topic_detection_scores` table has 18 records with sliding window detection data
2. The `manual_index_scores` table has 0 records (empty)
3. The code attempts to store window detection scores in BOTH tables:
   - General scores in `topic_detection_scores` (working)
   - Detailed window info in `manual_index_scores` (not working)

## The Problem
In `conversation.py` lines 522-553, the code tries to call `store_window_score()` which is imported as:
```python
from episodic.db_wrappers import store_topic_detection_score as store_window_score
```

This function expects window-specific parameters like:
- `window_a_start_short_id`
- `window_a_end_short_id`
- `window_b_start_short_id`
- `window_b_end_short_id`

## What's Happening
1. The sliding window detection IS working and creating topics correctly
2. The drift scores ARE being stored in `topic_detection_scores` table
3. The detailed window information is NOT being stored in `manual_index_scores` due to the function call failing silently
4. The application has a separate runtime error (Errno 22) that's preventing normal operation

## User's Concern
"The table topic_detection_scores has almost nothing in it" - Actually it has 18 records, but the user expected more detailed window information which should be in `manual_index_scores` but isn't.

## Next Steps
1. Fix the runtime error first (Errno 22)
2. Ensure window detection scores are properly stored in `manual_index_scores`
3. Verify both tables are being populated correctly