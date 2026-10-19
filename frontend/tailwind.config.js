/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: { colors: { paper: "#f5f4f0", ink: "#20211f", data: "#316fca", stress: "#c64c43" } } },
  plugins: [],
};
