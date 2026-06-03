const express = require('express');
const path    = require('path');
const { exec } = require('child_process');

const app  = express();
const PORT = 3000;

app.use(express.static(path.join(__dirname, 'public')));

app.get('*', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    const url = `http://localhost:${PORT}`;
    console.log(`\nSnake DQN frontend  ->  ${url}\n`);
    console.log('Make sure the Python backend is running: python server.py\n');
    // Auto-open browser on Windows
    exec(`start ${url}`, (err) => {
        if (err) console.log(`Open your browser at: ${url}`);
    });
});
