/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: "#6366f1",
                accent: "#a855f7",
                background: "#020617",
                card: "rgba(30, 41, 59, 0.4)",
            },
            fontFamily: {
                outfit: ['Outfit', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
