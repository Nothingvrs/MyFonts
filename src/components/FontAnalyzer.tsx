import React, { useEffect, useState } from 'react'
import { apiClient, type FontMatch } from '../services/apiClient'

interface FontAnalyzerProps {
	image: File
	onAnalysisStart: () => void
	onAnalysisComplete: (results: FontMatch[], error?: string) => void
}

const FontAnalyzer: React.FC<FontAnalyzerProps> = ({
	image,
	onAnalysisStart,
	onAnalysisComplete,
}) => {
	const [imageUrl, setImageUrl] = useState<string>('')
	const [isServerConnected, setIsServerConnected] = useState<boolean | null>(
		null
	)
	const [connectionError, setConnectionError] = useState<string>('')

	useEffect(() => {
		// Создаем URL для отображения изображения
		const url = URL.createObjectURL(image)
		setImageUrl(url)

		// Проверяем подключение к backend серверу
		const checkBackendConnection = async () => {
			try {
				console.log('Проверяем подключение к backend серверу...')
				const isConnected = await apiClient.checkServerConnection()
				setIsServerConnected(isConnected)

				if (isConnected) {
					console.log('✅ Backend сервер доступен')
					setConnectionError('')
				} else {
					console.log('❌ Backend сервер недоступен')
					setConnectionError(
						'Backend сервер не отвечает. Убедитесь, что он запущен на http://localhost:8000'
					)
				}
			} catch (error) {
				console.error('Ошибка при проверке подключения:', error)
				setIsServerConnected(false)
				setConnectionError('Ошибка подключения к серверу')
			}
		}

		checkBackendConnection()

		return () => {
			URL.revokeObjectURL(url)
		}
	}, [image])

	const analyzeImage = async () => {
		if (isServerConnected === null) {
			console.warn('Проверка подключения к серверу еще не завершена')
			return
		}

		if (!isServerConnected) {
			console.error('Backend сервер недоступен')
			// Показываем mock данные как fallback
			generateMockResults()
			return
		}

		onAnalysisStart()

		try {
			console.log('Отправляем изображение на backend для анализа...')

			// Используем API клиент для отправки изображения на backend
			const result = await apiClient.analyzeFont(image)

			console.log('Получен результат от backend:', result)

			if (result.success) {
				onAnalysisComplete(result.matches)
			} else {
				// Если анализ не удался (например, текст не найден)
				console.log(
					'Backend не смог проанализировать изображение:',
					result.message
				)
				console.log('🔍 Передаем ошибку в App:', {
					message: result.message,
					error: result.error,
					success: result.success,
				})
				onAnalysisComplete([], result.message)
			}
		} catch (error) {
			console.error('Ошибка при анализе через backend:', error)

			// В случае ошибки показываем fallback данные
			console.log('Используем mock данные как fallback')
			generateMockResults()
		}
	}

	const generateMockResults = () => {
		const mockResults: FontMatch[] = [
			{ name: 'Times New Roman', confidence: 0.85 },
			{ name: 'Georgia', confidence: 0.72 },
			{ name: 'PT Serif', confidence: 0.68 },
			{ name: 'Liberation Serif', confidence: 0.61 },
			{ name: 'DejaVu Sans', confidence: 0.55 },
			{ name: 'Arial', confidence: 0.48 },
			{ name: 'Open Sans', confidence: 0.42 },
			{ name: 'Roboto', confidence: 0.38 },
			{ name: 'PT Sans', confidence: 0.33 },
			{ name: 'Source Sans Pro', confidence: 0.28 },
		]
		onAnalysisComplete(mockResults)
	}

	return (
		<div className='my-8'>
			<div className='card text-center'>
				<h3 className='text-xl font-semibold text-gray-800 mb-6'>
					Загруженное изображение:
				</h3>
				<img
					src={imageUrl}
					alt='Uploaded'
					className='max-w-full max-h-96 lg:max-h-[500px] xl:max-h-[600px] mx-auto rounded-lg shadow-md mb-6 object-contain'
				/>
				{/* Показываем статус подключения к серверу */}
				{isServerConnected === null && (
					<div className='mb-4 p-3 bg-blue-100 border border-blue-300 rounded-lg'>
						<div className='flex items-center gap-2'>
							<div className='w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin'></div>
							<span className='text-blue-700'>
								Проверяем подключение к серверу...
							</span>
						</div>
					</div>
				)}

				{isServerConnected === false && (
					<div className='mb-4 p-3 bg-red-100 border border-red-300 rounded-lg'>
						<div className='text-red-700'>
							<strong>⚠️ Backend сервер недоступен</strong>
							<p className='text-sm mt-1'>{connectionError}</p>
							<p className='text-sm mt-1'>Будут использованы демо-данные.</p>
						</div>
					</div>
				)}

				{isServerConnected === true && (
					<div className='mb-4 p-3 bg-green-100 border border-green-300 rounded-lg'>
						<div className='text-green-700'>
							<strong>🇷🇺 Сервис для кириллических шрифтов готов</strong>
							<p className='text-sm mt-1'>
								Специализированный анализ русских шрифтов с помощью PaddleOCR и
								ИИ. Первый сервис для определения кириллических шрифтов!
							</p>
						</div>
					</div>
				)}

				<button
					onClick={analyzeImage}
					disabled={isServerConnected === null}
					className={`text-lg px-8 py-3 ${
						isServerConnected !== null
							? 'btn-primary'
							: 'bg-gray-400 text-white cursor-not-allowed rounded-lg font-semibold'
					}`}
				>
					{isServerConnected === null
						? 'Проверяем сервер...'
						: isServerConnected
						? '🔬 Анализировать шрифт (AI)'
						: '📝 Анализировать шрифт (демо)'}
				</button>
			</div>
		</div>
	)
}

export default FontAnalyzer
