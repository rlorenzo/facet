# UI Validation Testing Workflow

Patterns for systematically testing form validation, visual state, and UI flows.

## Complete Validation Test

```
1. navigate_page -> target URL
2. wait_for -> page content loaded
3. take_snapshot -> find field UIDs
4. fill_form -> enter invalid data
5. take_snapshot -> verify invalid="true", error message, disabled submit
6. take_screenshot -> visual confirmation of error state
7. fill -> correct the value
8. take_snapshot -> verify invalid cleared, submit enabled
9. take_screenshot -> visual confirmation of fixed state
```

## Clear State Between Tests

```typescript
// Navigate away and back to reset
mcp__chrome-devtools__navigate_page({ url: "http://localhost:4200/" });
mcp__chrome-devtools__wait_for({ text: "Facet" });
mcp__chrome-devtools__navigate_page({ url: target_url });
```

## Snapshot vs Screenshot

- **Snapshot** (`take_snapshot`): Structural verification -- attributes, text content, element tree, UIDs. **Preferred for debugging.**
- **Screenshot** (`take_screenshot`): Visual verification -- colors, layout, styling. Use for visual regression.

```typescript
// Full page screenshot
mcp__chrome-devtools__take_screenshot({ fullPage: true });

// Element-specific screenshot
mcp__chrome-devtools__take_screenshot({ uid: "element_uid" });

// Save to file
mcp__chrome-devtools__take_screenshot({ filePath: "/tmp/debug-screenshot.png" });
```

## Toggle/Checkbox Debugging

```
1. take_snapshot -> Find toggle UID and current state
2. click({ uid: toggle_uid }) -> Toggle it
3. list_network_requests({ resourceTypes: ["fetch"] }) -> Check if API called
4. get_network_request({ reqid: N }) -> Verify payload (correct property, correct value)
5. click({ uid: toggle_uid }) -> Toggle again (second click test)
6. list_network_requests -> Verify second API call made
7. get_network_request -> Verify second payload correct
```

## Form Validation State Reference

| State | Field Attribute | Submit Button |
|:---:|:---:|:---:|
| Invalid | `invalid="true"` | `disableable disabled` |
| Valid | No `invalid` attr | No `disabled` attr |
