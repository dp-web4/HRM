# Coordinate-Aware Hybrid Architecture

## McNugget's Critical Discovery (2026-03-31)

**ACTION6 is a ComplexAction requiring x,y coordinates.**

```python
# ✗ WRONG (what we were doing):
env.step(GameAction.ACTION6)  # Does nothing!

# ✓ CORRECT (McNugget's discovery):
env.step(GameAction.ACTION6, data={'x': col, 'y': row})  # Clicks on grid[row, col]
```

This explains why 100k+ steps with ACTION6 achieved 0 levels across all previous approaches.

---

## Integration with Hybrid Architecture

### Before (sage_driver.py + sage_navigator.py)
- Navigator suggests: "Try ACTION6" (no coordinates)
- Driver executes: ACTION6 blindly
- Result: No effect

### After (coordinate-aware hybrid)
- Navigator suggests: "Click on bright cells in top-left quadrant"
- Driver translates to: Multiple clicks with (x,y) coordinates
- Membot stores: "Clicking color 14 at row<10 completed level 1"

---

## Updated Navigator Prompt

**Old prompt:**
```
Which actions should we try next?
Response: {"action_bias": {"UP": 0.3, "ACTION6": 0.7}}
```

**New prompt:**
```
Which actions and WHERE should we try next?
Response: {
  "action_bias": {"UP": 0.2, "CLICK": 0.8},
  "click_strategy": {
    "target_colors": [14, 15],
    "target_regions": ["top-left", "columns-62-63"],
    "priority": "color_boundaries"
  }
}
```

---

## Updated Driver Action Selection

```python
def select_action_with_coords(self, state_hash: str, grid: np.ndarray) -> tuple:
    """Select action + coordinates if applicable.

    Returns:
        (action_int, coords_dict or None)
    """
    action = self.select_action(state_hash, grid)  # Existing logic

    if action == 6 and self.llm_bias:  # ACTION6 requires coordinates
        click_strategy = self.llm_bias.get("click_strategy", {})
        coords = self._select_click_target(grid, click_strategy)
        return (action, coords)
    else:
        return (action, None)

def _select_click_target(self, grid: np.ndarray, strategy: dict) -> dict:
    """Select x,y coordinates for ACTION6 click.

    Args:
        grid: Current grid state
        strategy: LLM's click strategy from Navigator

    Returns:
        {'x': col, 'y': row}
    """
    target_colors = strategy.get("target_colors", [])
    target_regions = strategy.get("target_regions", [])
    priority = strategy.get("priority", "any")

    # Find all non-background cells
    bg = int(np.bincount(grid.flatten()).argmax())
    candidates = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == bg:
                continue

            # Filter by target colors if specified
            if target_colors and color not in target_colors:
                continue

            # Filter by region if specified
            if "top-left" in target_regions and (r > grid.shape[0]//2 or c > grid.shape[1]//2):
                continue
            if "columns-62-63" in target_regions and c not in [62, 63]:
                continue

            candidates.append((r, c, color))

    if not candidates:
        # Fallback: click center
        h, w = grid.shape
        return {'x': w//2, 'y': h//2}

    # Select based on priority
    if priority == "color_boundaries":
        # Prefer cells adjacent to different colors
        scored = []
        for r, c, color in candidates:
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    neighbors.append(int(grid[nr, nc]))
            unique_neighbors = len(set(neighbors))
            scored.append((unique_neighbors, r, c, color))
        scored.sort(reverse=True)
        _, r, c, _ = scored[0]
        return {'x': c, 'y': r}
    else:
        # Random from candidates
        r, c, _ = random.choice(candidates)
        return {'x': c, 'y': r}
```

---

## Updated Game Loop (sage_hybrid_runner.py)

```python
while step < max_steps:
    state_hash = grid_hash(grid)
    action, coords = driver.select_action_with_coords(state_hash, grid)

    # Execute action
    if coords:
        frame_data = env.step(INT_TO_GAME_ACTION[action], data=coords)
        action_desc = f"CLICK({coords['y']},{coords['x']}) color={int(grid[coords['y'], coords['x']])}"
    else:
        frame_data = env.step(INT_TO_GAME_ACTION[action])
        action_desc = ACTION_LABELS.get(action, f"A{action}")

    # Record outcome
    changed = not np.array_equal(prev_grid, grid)
    driver.record(action, state_hash, changed, coords=coords)

    # Navigator reflection every 20 actions
    if actions_since_reflection >= 20:
        reflection = navigator.reflect_with_grid(
            driver_history=driver.get_recent_history(),
            current_grid=grid,
            levels_completed=frame_data.levels_completed
        )
        driver.update_llm_bias(reflection)
        actions_since_reflection = 0
```

---

## Membot Cartridge Updates

**Store coordinate patterns that complete levels:**

```json
{
  "winning_sequences": [
    {
      "level": 1,
      "actions": ["UP", "UP", "UP", "CLICK(0,62)", "CLICK(0,63)", ...],
      "click_patterns": [
        {"color": 14, "region": "top-right", "effectiveness": 1.0},
        {"color": 0, "region": "columns-62-63", "effectiveness": 0.8}
      ]
    }
  ],
  "click_effectiveness": {
    "global": {
      "color_14": {"tries": 50, "changes": 45, "rate": 0.9},
      "color_0": {"tries": 30, "changes": 24, "rate": 0.8}
    },
    "by_region": {
      "top-left": {"tries": 100, "level_ups": 2},
      "columns-62-63": {"tries": 80, "level_ups": 1}
    }
  }
}
```

---

## Performance Expectations

**Without coordinates:**
- 100k steps, 0 levels (McNugget sage_learner.py)
- 5M steps, 0 levels (prior exploration)

**With coordinates (clicker):**
- 5k steps, 0 levels (blind systematic clicking)
- But clicks DO cause grid changes!

**With coordinates + LLM guidance (hybrid):**
- Target: First level completion in <10k steps
- Hypothesis: Navigator identifies goal patterns → guides clicks to goal-relevant cells → levels complete

---

## Next Steps

1. **Update sage_navigator.py** - Add click_strategy to reflection output
2. **Update sage_driver.py** - Add `select_action_with_coords()` and `_select_click_target()`
3. **Build sage_hybrid_runner.py** - Integrate coordinate-aware actions
4. **Test on sc25** - Known to have ACTION6 (McNugget's logs show color clicks work)
5. **First score** - Document which coordinate patterns complete levels

---

*Integration plan: 2026-03-31*
*Builds on: McNugget's sage_clicker.py discovery + Thor's hybrid architecture*
