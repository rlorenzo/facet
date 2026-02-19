# evaluate_script Patterns

Detailed patterns for using `evaluate_script` to inspect and manipulate the DOM directly.

## Component State Inspection

```typescript
// Check Angular component properties via DOM
mcp__chrome-devtools__evaluate_script({
    function: `() => {
        const form = document.querySelector('form');
        const inputs = Array.from(form.querySelectorAll('input'));
        return {
            formDisabled: form?.disabled,
            invalidInputs: inputs.filter(i => !i.validity.valid).length,
            formClasses: form?.className,
            inputValues: inputs.slice(0, 5).map(i => ({ name: i.name, value: i.value, valid: i.validity.valid }))
        };
    }`
});
```

## Simulate User Input (Precise)

Use when `fill()` doesn't trigger Angular change detection properly:

```typescript
mcp__chrome-devtools__evaluate_script({
    function: `() => {
        const input = document.querySelector('input[type="number"]');
        input.focus();
        input.select();
        input.value = '9';
        input.dispatchEvent(new InputEvent('input', { bubbles: true }));
        return { newValue: input.value };
    }`
});
```

## With Element Arguments

```typescript
// Pass element UIDs to evaluate_script
mcp__chrome-devtools__evaluate_script({
    function: `(el) => { return el.innerText; }`,
    args: [{ uid: "element_uid" }]
});
```

## Value Revert Detection

When user reports input value reverting after entry:

```typescript
// 1. Navigate and take snapshot
mcp__chrome-devtools__navigate_page({ url: "http://localhost:4200/settings" });
mcp__chrome-devtools__take_snapshot();

// 2. Change value via evaluate_script for precise control
mcp__chrome-devtools__evaluate_script({
    function: `() => {
        const inputs = document.querySelectorAll('input[type="number"]');
        inputs[0].focus();
        inputs[0].select();
        inputs[0].value = '9';
        inputs[0].dispatchEvent(new InputEvent('input', { bubbles: true }));
        return { changed: true, value: inputs[0].value };
    }`
});

// 3. Check if value persists after change detection
mcp__chrome-devtools__evaluate_script({
    function: `() => {
        const inputs = document.querySelectorAll('input[type="number"]');
        return {
            value: inputs[0].value,
            persisted: inputs[0].value === '9'
        };
    }`
});
// persisted: false -> effect or signal reset destroying user edits
```

## Multiple Component Instance Detection

```typescript
mcp__chrome-devtools__evaluate_script({
    function: `() => {
        const components = document.querySelectorAll('app-photo-card');
        return {
            count: components.length,
            issue: components.length > 100
                ? 'Excessive instances - check virtual scrolling or pagination'
                : 'Count OK'
        };
    }`
});
```
