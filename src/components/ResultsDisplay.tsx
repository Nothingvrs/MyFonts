import React, { useEffect } from 'react'
import type { FontMatch } from '../services/apiClient'

interface ResultsDisplayProps {
	results: FontMatch[]
	errorMessage?: string | null
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
	results,
	errorMessage,
}) => {
	// Функция для загрузки шрифтов из Google Fonts
	const loadGoogleFonts = (fontNames: string[]) => {
		// Удаляем существующие ссылки на шрифты
		const existingLinks = document.querySelectorAll('link[data-font-loader]')
		existingLinks.forEach(link => link.remove())

		if (fontNames.length === 0) return

		// Формируем URL для Google Fonts API
		const fontQuery = fontNames.map(name => name.replace(/\s+/g, '+')).join('|')

		const link = document.createElement('link')
		link.href = `https://fonts.googleapis.com/css2?family=${fontQuery}&display=swap`
		link.rel = 'stylesheet'
		link.setAttribute('data-font-loader', 'true')
		document.head.appendChild(link)
	}

	// Загружаем шрифты при изменении результатов
	useEffect(() => {
		if (results.length > 0) {
			const fontNames = results.map(result => result.name)
			loadGoogleFonts(fontNames)
		}
	}, [results])

	const getConfidenceColor = (confidence: number): string => {
		if (confidence >= 0.8) return '#28a745' // Зеленый
		if (confidence >= 0.6) return '#ffc107' // Желтый
		if (confidence >= 0.4) return '#fd7e14' // Оранжевый
		return '#dc3545' // Красный
	}

	const getConfidenceText = (confidence: number): string => {
		if (confidence >= 0.8) return 'Высокая'
		if (confidence >= 0.6) return 'Средняя'
		if (confidence >= 0.4) return 'Низкая'
		return 'Очень низкая'
	}

	// ПРИОРИТЕТ 1: Если есть сообщение об ошибке, показываем его (важнее чем просто "нет результатов")
	if (errorMessage) {
		// Определяем тип ошибки для показа соответствующего UI (более надежно)
		const isMultipleFonts =
			errorMessage.includes('несколько разных шрифтов') ||
			errorMessage.includes('несколько') ||
			errorMessage.includes('множественные') ||
			errorMessage.toLowerCase().includes('multiple')

		const isNoText =
			errorMessage.includes('не обнаружен текст') ||
			errorMessage.includes('не найден текст') ||
			errorMessage.toLowerCase().includes('no text')

		const isOcrError =
			errorMessage.includes('распознавания текста') ||
			errorMessage.toLowerCase().includes('ocr')

		return (
			<div className='my-8'>
				<div className='card p-8 text-center'>
					<div
						className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${
							isMultipleFonts
								? 'bg-orange-100'
								: isOcrError
								? 'bg-yellow-100'
								: 'bg-red-100'
						}`}
					>
						<span className='text-3xl'>
							{isMultipleFonts ? '🔤' : isOcrError ? '🔧' : '⚠️'}
						</span>
					</div>
					<h3
						className={`text-xl font-semibold mb-2 ${
							isMultipleFonts
								? 'text-orange-600'
								: isOcrError
								? 'text-yellow-600'
								: 'text-red-600'
						}`}
					>
						{isMultipleFonts
							? 'Обнаружено несколько шрифтов'
							: isOcrError
							? 'Ошибка распознавания'
							: 'Текст не обнаружен'}
					</h3>
					<p className='text-gray-600 mb-4'>{errorMessage}</p>
					<div className='bg-blue-50 border border-blue-200 rounded-lg p-4 text-left max-w-md mx-auto'>
						<h4 className='font-medium text-blue-800 mb-2'>💡 Рекомендации:</h4>
						{isMultipleFonts ? (
							<ul className='text-sm text-blue-700 space-y-1'>
								<li>• Обрежьте изображение, оставив только один шрифт</li>
								<li>• Используйте изображение с однородным текстом</li>
								<li>• Избегайте заголовков с разными шрифтами</li>
								<li>• Загрузите отдельные изображения для каждого шрифта</li>
							</ul>
						) : isOcrError ? (
							<ul className='text-sm text-blue-700 space-y-1'>
								<li>• Попробуйте другое изображение</li>
								<li>• Убедитесь, что изображение не повреждено</li>
								<li>• Используйте изображения в формате PNG или JPG</li>
								<li>• Проверьте, что изображение содержит текст</li>
							</ul>
						) : (
							<ul className='text-sm text-blue-700 space-y-1'>
								<li>• Убедитесь, что изображение содержит четкий текст</li>
								<li>• Попробуйте изображение с более контрастным текстом</li>
								<li>• Используйте изображения с кириллическими символами</li>
								<li>• Избегайте слишком мелкого или размытого текста</li>
							</ul>
						)}
					</div>
				</div>
			</div>
		)
	}

	// ПРИОРИТЕТ 2: Если результатов нет (и нет сообщения об ошибке), показываем общее сообщение
	if (results.length === 0) {
		return (
			<div className='my-8'>
				<div className='card text-center py-12'>
					<div className='text-6xl mb-4'>🔍</div>
					<h2 className='text-2xl font-bold text-gray-800 mb-4'>
						Текст не обнаружен
					</h2>
					<p className='text-gray-600 max-w-md mx-auto mb-6'>
						На загруженном изображении не удалось найти текст для анализа.
						Попробуйте загрузить изображение с четким, читаемым текстом.
					</p>
					<div className='text-sm text-gray-500'>
						<p>
							<strong>Рекомендации:</strong>
						</p>
						<ul className='list-disc list-inside mt-2 space-y-1'>
							<li>Убедитесь, что текст контрастный и хорошо виден</li>
							<li>Избегайте изображений только с графикой или фотографиями</li>
							<li>Текст должен быть достаточно крупным</li>
						</ul>
					</div>
				</div>
			</div>
		)
	}

	return (
		<div className='my-8'>
			<div className='text-center mb-8'>
				<h2 className='text-3xl font-bold text-white mb-4'>
					Результаты анализа
				</h2>
				<p className='text-lg text-white/80 max-w-2xl mx-auto'>
					Найдено {results.length} похожих шрифтов, отсортированных по степени
					совпадения:
				</p>
			</div>

			<div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6'>
				{results.map((result, index) => (
					<div
						key={result.font_info?.id || index}
						className='font-card hover:shadow-xl transition-all duration-300 hover:-translate-y-1'
					>
						<div className='flex items-center gap-3 mb-4'>
							<span className='bg-blue-500 text-white text-sm font-bold px-3 py-1 rounded-full'>
								#{index + 1}
							</span>
							<h3 className='text-lg font-semibold text-gray-800 flex-1'>
								{result.name}
							</h3>
							{result.font_info?.cyrillic_support && (
								<span
									className='text-blue-500 text-sm'
									title='Поддержка кириллицы'
								>
									🇷🇺
								</span>
							)}
						</div>

						{/* Дополнительная информация о шрифте */}
						{result.font_info && (
							<div className='mb-3 text-xs text-gray-600 space-y-1'>
								<div className='flex justify-between'>
									<span>Категория:</span>
									<span className='font-medium capitalize'>
										{result.font_info.category}
									</span>
								</div>
								{result.font_info.designer && (
									<div className='flex justify-between'>
										<span>Дизайнер:</span>
										<span className='font-medium text-right'>
											{result.font_info.designer}
										</span>
									</div>
								)}
								{result.font_info.year && (
									<div className='flex justify-between'>
										<span>Год:</span>
										<span className='font-medium'>{result.font_info.year}</span>
									</div>
								)}
							</div>
						)}

						<div className='mb-4'>
							<div className='bg-gray-200 rounded-full h-2 overflow-hidden mb-2'>
								<div
									className='h-full rounded-full transition-all duration-500'
									style={{
										width: `${result.confidence * 100}%`,
										backgroundColor: getConfidenceColor(result.confidence),
									}}
								/>
							</div>
							<div className='flex justify-between items-center text-sm'>
								<span className='font-semibold text-gray-700'>
									{Math.round(result.confidence * 100)}%
								</span>
								<span
									className='font-medium'
									style={{ color: getConfidenceColor(result.confidence) }}
								>
									{getConfidenceText(result.confidence)}
								</span>
							</div>
						</div>

						<div className='bg-gray-50 rounded-lg p-4 mb-4 text-center'>
							<p className='text-sm text-gray-600 mb-2'>Пример кириллицы:</p>
							<div
								className='font-preview text-lg text-gray-800 leading-tight'
								style={{ fontFamily: `'${result.name}', serif` }}
								title='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
							>
								АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
							</div>
						</div>

						<div className='space-y-2'>
							<div className='flex gap-2'>
								<a
									href={`https://fonts.google.com/specimen/${result.name.replace(
										/\s+/g,
										'+'
									)}`}
									target='_blank'
									rel='noopener noreferrer'
									className='btn-primary flex-1 text-sm py-2 text-center no-underline hover:no-underline'
								>
									Google Fonts
								</a>
								<a
									href={`https://fonts.adobe.com/fonts/${result.name
										.toLowerCase()
										.replace(/\s+/g, '-')}`}
									target='_blank'
									rel='noopener noreferrer'
									className='btn-secondary flex-1 text-sm py-2 text-center no-underline hover:no-underline'
								>
									Adobe Fonts
								</a>
							</div>
							<div className='flex gap-2'>
								<a
									href={`https://www.fontsquirrel.com/fonts/${result.name
										.toLowerCase()
										.replace(/\s+/g, '-')}`}
									target='_blank'
									rel='noopener noreferrer'
									className='btn-secondary flex-1 text-xs py-1.5 text-center no-underline hover:no-underline'
								>
									Font Squirrel
								</a>
								<a
									href={`https://www.dafont.com/search.php?q=${result.name.replace(
										/\s+/g,
										'+'
									)}`}
									target='_blank'
									rel='noopener noreferrer'
									className='btn-secondary flex-1 text-xs py-1.5 text-center no-underline hover:no-underline'
								>
									DaFont
								</a>
								<a
									href={`https://www.myfonts.com/search/${result.name.replace(
										/\s+/g,
										'+'
									)}/fonts/`}
									target='_blank'
									rel='noopener noreferrer'
									className='btn-secondary flex-1 text-xs py-1.5 text-center no-underline hover:no-underline'
								>
									MyFonts
								</a>
							</div>
						</div>
					</div>
				))}
			</div>
		</div>
	)
}

export default ResultsDisplay
