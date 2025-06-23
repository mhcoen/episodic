# Test Scripts for Topic Detection

This directory contains various test scripts for validating the topic detection and extraction system.

## Usage

Run any script with:
```
/script scripts/filename.txt
```

After running, check results with:
```
/topics
```

## Test Scripts

### basic-topic-changes.txt
Tests clear, obvious topic shifts across different domains:
- Movies → Quantum Physics → Baseball → Programming → Cooking
- Should trigger mostly "high confidence" topic changes

### gradual-drift.txt  
Tests subtle topic transitions that flow naturally:
- Science Fiction → AI → Quantum Computing → Physics → Biology
- Should trigger "medium" or "low" confidence changes, or possibly no changes

### false-positive-test.txt
Tests conversation that should NOT trigger topic changes:
- All about semantic drift and topic detection
- Should show minimal or no topic change indicators

### rapid-switching.txt
Tests quick topic changes between unrelated subjects:
- Weather → Physics → Cooking → AI → Sports → etc.
- Should trigger mostly "high confidence" changes

### edge-cases.txt
Tests unusual scenarios:
- Greetings, philosophical questions, humor requests
- Mixed expectations for topic change detection

### meta-conversation.txt
Tests discussion about the system itself mixed with other topics:
- System discussion → Space → System discussion → Food
- Interesting test of how LLM handles meta-conversations

## Expected Outcomes

- **High confidence**: Clear domain shifts (movies → sports)
- **Medium confidence**: Related but different focus (physics → chemistry)  
- **Low confidence**: Subtle shifts within domain (sci-fi movies → AI movies)
- **No change**: Natural follow-ups and elaborations

## Tips

- Run `/set debug true` to see detailed topic extraction info
- Clear topics between tests if needed
- Compare topic extraction quality across different scripts