/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { paper: "#f4f4f0", surface: "#ffffff", ink: "#17181c", data: "#3b4fd6", breach: "#ad4a2c" },
      fontFamily: { mono: ['"Space Mono"', "monospace"], sans: ['"Space Grotesk"', "sans-serif"] },
    },
  },
  plugins: [],
};
