/**
 * API клиент для взаимодействия с backend
 */

export interface FontMatch {
	name: string
	confidence: number
	preview?: string
	font_info?: FontInfo
	match_details?: {
		overall_score: number
		serif_match: number
		stroke_match: number
		contrast_match: number
		cyrillic_match: number
		geometric_match: number
		spacing_match: number
	}
}

export type FontCategory =
	| 'serif'
	| 'sans-serif'
	| 'monospace'
	| 'display'
	| 'handwriting'
	| 'script'

export interface CyrillicFeatures {
	ya_shape: number
	zh_shape: number
	fi_shape: number
	shcha_shape: number
	yery_shape: number
}

export interface FontCharacteristics {
	has_serifs: boolean
	stroke_width: number
	contrast: number
	slant: number
	cyrillic_features: CyrillicFeatures
	x_height: number
	cap_height: number
	ascender: number
	descender: number
	letter_spacing: number
	word_spacing: number
	density: number
}

export interface FontInfo {
	id?: number
	name: string
	category: FontCategory
	characteristics: FontCharacteristics
	popularity: number
	cyrillic_support: boolean
	designer?: string
	year?: number
	foundry?: string
	description?: string
	license?: string
	download_url?: string
}

export interface AnalysisResult {
	success: boolean
	message: string
	matches: FontMatch[]
	characteristics?: FontCharacteristics
	error?: string
	processing_time?: number
}

export interface ApiError {
	error: string
	message: string
	details?: unknown
}

class ApiClient {
	private baseUrl: string

	constructor() {
		// URL backend сервера
		const envBase = (import.meta as any).env?.VITE_API_BASE_URL as
			| string
			| undefined
		this.baseUrl =
			envBase && envBase.trim().length > 0 ? envBase : 'http://localhost:8000'
	}

	/**
	 * Анализ шрифта по изображению
	 */
	async analyzeFont(imageFile: File): Promise<AnalysisResult> {
		try {
			console.log('Отправляем изображение на анализ...', {
				name: imageFile.name,
				size: imageFile.size,
				type: imageFile.type,
			})

			const formData = new FormData()
			formData.append('file', imageFile)

			const response = await fetch(`${this.baseUrl}/api/analyze-font`, {
				method: 'POST',
				body: formData,
			})

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}))
				throw new Error(
					errorData.message || `HTTP ${response.status}: ${response.statusText}`
				)
			}

			const result: AnalysisResult = await response.json()

			console.log('Получен результат анализа:', result)

			// Преобразуем результат для совместимости с фронтендом
			const transformedResult: AnalysisResult = {
				...result,
				matches: result.matches.map(match => ({
					name: match.font_info?.name || match.name,
					confidence: match.confidence,
					preview: match.font_info?.name || match.name,
					font_info: match.font_info,
					match_details: match.match_details,
				})),
			}

			return transformedResult
		} catch (error) {
			console.error('Ошибка при анализе шрифта:', error)

			// Проверяем, связана ли ошибка с подключением к серверу
			if (error instanceof TypeError && error.message.includes('fetch')) {
				throw new Error(
					'Не удалось подключиться к серверу. Убедитесь, что backend запущен на http://localhost:8000'
				)
			}

			throw error
		}
	}

	/**
	 * Получение списка всех шрифтов
	 */
	async getFonts(category?: string): Promise<FontInfo[]> {
		try {
			const url = category
				? `${this.baseUrl}/api/fonts?category=${encodeURIComponent(category)}`
				: `${this.baseUrl}/api/fonts`

			const response = await fetch(url)

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`)
			}

			return await response.json()
		} catch (error) {
			console.error('Ошибка при получении списка шрифтов:', error)
			throw error
		}
	}

	/**
	 * Получение информации о шрифте по ID
	 */
	async getFontById(fontId: number): Promise<FontInfo> {
		try {
			const response = await fetch(`${this.baseUrl}/api/fonts/${fontId}`)

			if (!response.ok) {
				if (response.status === 404) {
					throw new Error('Шрифт не найден')
				}
				throw new Error(`HTTP ${response.status}: ${response.statusText}`)
			}

			return await response.json()
		} catch (error) {
			console.error(`Ошибка при получении шрифта ${fontId}:`, error)
			throw error
		}
	}

	/**
	 * Поиск шрифтов по запросу
	 */
	async searchFonts(query: string): Promise<FontInfo[]> {
		try {
			const response = await fetch(
				`${this.baseUrl}/api/fonts/search/${encodeURIComponent(query)}`
			)

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`)
			}

			return await response.json()
		} catch (error) {
			console.error(`Ошибка при поиске шрифтов "${query}":`, error)
			throw error
		}
	}

	/**
	 * Проверка здоровья API
	 */
	async healthCheck(): Promise<{ status: string; message: string }> {
		try {
			const response = await fetch(`${this.baseUrl}/api/health`)

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`)
			}

			return await response.json()
		} catch (error) {
			console.error('Ошибка при проверке здоровья API:', error)
			throw error
		}
	}

	/**
	 * Проверка доступности backend сервера
	 */
	async checkServerConnection(): Promise<boolean> {
		try {
			await this.healthCheck()
			return true
		} catch {
			return false
		}
	}
}

// Экспортируем единственный экземпляр
export const apiClient = new ApiClient()
export default apiClient
