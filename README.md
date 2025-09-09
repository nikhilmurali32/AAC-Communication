# AAC Frontend

This is a React-based front end for an Augmentative and Alternative Communication (AAC) system.

## Features
- Clickable, colorful word boxes for common AAC vocabulary
- Search box at the bottom for sentence construction
- Send button to submit input
- Translucent loading/output screen with close (X) button
- Easy integration point for backend model (Flan-T5 or other)
<img width="1916" height="990" alt="image" src="https://github.com/user-attachments/assets/41fae526-de4a-4e85-86b2-ca4752810820" />

## How to Run

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm start
   ```

The app will be available at `http://localhost:3000` by default.

## Backend Integration

- In `src/App.js`, look for the following comment:
  ```js
  // TODO: Replace this timeout with actual backend call
  // Example: fetch('/api/your-backend-endpoint', { method: 'POST', body: JSON.stringify({ input }) })
  ```
- Replace the `setTimeout` block with your actual backend call (e.g., using `fetch` or `axios`).
- When you receive the response from your backend (Flan-T5 model), set the output using `setOutput(responseText)`.

<img width="587" height="394" alt="image" src="https://github.com/user-attachments/assets/ab3e995f-9b63-44c3-8160-fd0bbd3fa84c" />


## Example Backend Call
```js
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input })
})
  .then(res => res.json())
  .then(data => {
    setOutput(data.output); // Adjust according to your backend response
    setLoading(false);
  })
  .catch(() => setLoading(false));
```
<img width="317" height="321" alt="image" src="https://github.com/user-attachments/assets/fdb77218-431f-4111-9818-a833b3448117" />

## Customization
- To change the words or colors, edit the `WORDS` array in `src/App.js`.
- To change the look and feel, adjust the inline styles in `src/App.js`. 

