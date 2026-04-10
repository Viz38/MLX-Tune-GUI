# MLX-Tune GUI

The professional, Mac-native orchestration interface for `mlx-tune`.

## Getting Started

First, install the dependencies:

```bash
cd web
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Architecture

- **Framework**: Next.js 15+ (App Router)
- **Styling**: Deep Dark Mac-native CSS (Vanilla CSS)
- **State**: React Hooks + LocalStorage Persistence
- **API**: Next.js Route Handlers for shell execution and Finder integration

## Development

- `src/app/page.tsx`: Main orchestration logic and UI.
- `src/app/globals.css`: Custom "Deep Dark" design system.
- `src/lib/scriptGenerator.ts`: Maps UI state to executable Python scripts.
- `src/app/api/run/route.ts`: Streams training logs from the shell to the UI.
