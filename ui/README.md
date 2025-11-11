# NPU Chat UI

A modern React-based chat interface for the OpenVINO GenAI NPU server.

## Features

- 💬 Chat with multiple AI models
- 🖼️ Image input support (VLM models)
- 🔊 Audio output support (TTS)
- 📱 Responsive design
- 💾 Chat history persistence
- 🎨 Dark theme UI

## Prerequisites

- Node.js 18+ and npm/yarn
- NPU server running on `http://localhost:8000`

## Installation

1. Navigate to the UI directory:
```bash
cd ui
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

## Development

Start the development server:

```bash
npm run dev
# or
yarn dev
```

The UI will be available at `http://localhost:3000`.

The development server is configured to proxy API requests to the NPU server at `http://localhost:8000`.

## Building for Production

Build the production bundle:

```bash
npm run build
# or
yarn build
```

The built files will be in the `dist` directory.

Preview the production build:

```bash
npm run preview
# or
yarn preview
```

## Configuration

The API base URL can be configured via environment variable:

- Create a `.env` file in the `ui` directory
- Add: `VITE_API_URL=http://localhost:8000`
- Default: `http://localhost:8000`

## Project Structure

```
ui/
├── src/
│   ├── components/      # React components
│   ├── contexts/        # React contexts
│   ├── hooks/           # Custom hooks
│   ├── services/        # API services
│   ├── styles/          # CSS styles
│   └── utils/           # Utility functions
├── index.html
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## Technologies

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **Lucide React** - Icons

