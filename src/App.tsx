import { useState } from 'react'
import FontAnalyzer from './components/FontAnalyzer'
import ImageUploader from './components/ImageUploader'
import LottieScanner from './components/LottieScanner'
import ResultsDisplay from './components/ResultsDisplay'

interface FontMatch {
	name: string
	confidence: number
	preview?: string
}

function App() {
	const [uploadedImage, setUploadedImage] = useState<File | null>(null)
	const [analysisResults, setAnalysisResults] = useState<FontMatch[]>([])
	const [isAnalyzing, setIsAnalyzing] = useState(false)
	const [errorMessage, setErrorMessage] = useState<string | null>(null)

	const handleImageUpload = (file: File) => {
		setUploadedImage(file)
		setAnalysisResults([])
		setErrorMessage(null)
	}

	const handleAnalysisStart = () => {
		setIsAnalyzing(true)
		setAnalysisResults([])
		setErrorMessage(null)
	}

	const handleAnalysisComplete = (results: FontMatch[], error?: string) => {
		console.log('🔍 App получил:', {
			results: results.length,
			error,
			errorType: typeof error,
		})
		setAnalysisResults(results)
		setIsAnalyzing(false)
		setErrorMessage(error || null)
	}

	return (
		<div className='min-h-screen gradient-bg'>
			<header className='bg-white/95 backdrop-blur-sm shadow-lg'>
				<div className='max-w-[1400px] mx-auto px-4 lg:px-8 xl:px-12 py-8 text-center'>
					<h1 className='text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4'>
						MyFonts - Определение кириллических шрифтов
					</h1>
					<p className='text-lg text-gray-600 max-w-2xl mx-auto'>
						Загрузите изображение с текстом, и мы поможем определить шрифт
					</p>
				</div>
			</header>

			<main className='max-w-[1600px] mx-auto px-4 lg:px-8 xl:px-12 py-8'>
				{!uploadedImage ? (
					<ImageUploader onImageUpload={handleImageUpload} />
				) : (
					<div className='animate-fade-in'>
						<div className='flex justify-end mb-4'>
							<button
								onClick={() => {
									if (isAnalyzing) return
									setUploadedImage(null)
									setAnalysisResults([])
									setErrorMessage(null)
									setIsAnalyzing(false)
								}}
								disabled={isAnalyzing}
								className={`text-sm px-4 py-2 rounded-lg font-medium ${
									isAnalyzing
										? 'bg-gray-300 text-gray-600 cursor-not-allowed'
										: 'btn-secondary'
								}`}
							>
								{isAnalyzing
									? 'Анализ выполняется…'
									: 'Загрузить другое изображение'}
							</button>
						</div>
						<FontAnalyzer
							image={uploadedImage}
							onAnalysisStart={handleAnalysisStart}
							onAnalysisComplete={handleAnalysisComplete}
						/>
					</div>
				)}

				{/* Во время анализа показываем Lottie над будущими результатами */}
				{isAnalyzing && (
					<div className='animate-fade-in mt-6'>
						{(() => {
							const base = (import.meta as any).env?.BASE_URL || '/'
							const lottieSrc = `${base.replace(
								/\/$/,
								'/'
							)}lottie/robot-scan.json`
							return (
								<LottieScanner
									playing={true}
									className='w-full'
									src={lottieSrc}
								/>
							)
						})()}
					</div>
				)}

				{(analysisResults.length > 0 || errorMessage) && (
					<div className='animate-slide-up mt-6'>
						<ResultsDisplay
							results={analysisResults}
							errorMessage={errorMessage}
						/>
					</div>
				)}
			</main>
		</div>
	)
}

export default App
