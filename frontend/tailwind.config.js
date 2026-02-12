/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                medical: {
                    50: '#f0f9f9',
                    100: '#d9f2f2',
                    200: '#b8e5e5',
                    300: '#8bd1d1',
                    400: '#5ab6b6',
                    500: '#3f9a9a',
                    600: '#347d7d',
                    700: '#2e6666',
                    800: '#295454',
                    900: '#264747',
                    950: '#112a2a',
                },
            },
            backgroundImage: {
                'glass-gradient': 'linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))',
            },
        },
    },
    plugins: [],
}
