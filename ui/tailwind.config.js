/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme colors inspired by modern chat UIs
        dark: {
          bg: '#0f0f0f',
          surface: '#1a1a1a',
          surfaceHover: '#252525',
          border: '#2a2a2a',
          text: '#e5e5e5',
          textSecondary: '#a0a0a0',
          accent: '#10b981',
          accentHover: '#059669',
          error: '#ef4444',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}

