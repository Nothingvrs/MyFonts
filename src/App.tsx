import { useState } from 'react'
import FontAnalyzer from './components/FontAnalyzer'
import ImageUploader from './components/ImageUploader'
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
				<ImageUploader onImageUpload={handleImageUpload} />

				{uploadedImage && (
					<div className='animate-fade-in'>
						<FontAnalyzer
							image={uploadedImage}
							onAnalysisStart={() => setIsAnalyzing(true)}
							onAnalysisComplete={handleAnalysisComplete}
						/>
					</div>
				)}

				{isAnalyzing && (
					<div className='card text-center my-8 animate-pulse'>
						<div className='flex items-center justify-center gap-3'>
							<div className='w-6 h-6 border-4 border-blue-500 border-t-transparent rounded-full animate-spin'></div>
							<p className='text-lg font-medium text-gray-700'>
								Анализируем изображение...
							</p>
						</div>
					</div>
				)}

				{(analysisResults.length > 0 || errorMessage) && (
					<div className='animate-slide-up'>
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
