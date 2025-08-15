// Временный сервер на Node.js
const http = require('http')
const fs = require('fs')
const path = require('path')

const server = http.createServer((req, res) => {
	// CORS headers
	res.setHeader('Access-Control-Allow-Origin', '*')
	res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
	res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

	if (req.method === 'OPTIONS') {
		res.writeHead(200)
		res.end()
		return
	}

	if (req.url === '/api/health') {
		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ status: 'healthy', message: 'Temp server OK' }))
		return
	}

	if (req.url === '/api/analyze-font' && req.method === 'POST') {
		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(
			JSON.stringify({
				success: false,
				message: 'Python сервер недоступен. Исправьте установку Python.',
				matches: [],
				error: 'PYTHON_ERROR',
			})
		)
		return
	}

	res.writeHead(200, { 'Content-Type': 'application/json' })
	res.end(JSON.stringify({ message: 'Temporary server running' }))
})

const PORT = 8000
server.listen(PORT, () => {
	console.log(`🚀 Временный сервер запущен на http://localhost:${PORT}`)
	console.log('❗ Это временное решение. Нужно исправить Python.')
})

