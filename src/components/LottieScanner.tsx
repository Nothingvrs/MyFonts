import lottie, { type AnimationItem } from 'lottie-web'
import React, { useEffect, useRef } from 'react'

interface LottieScannerProps {
	playing: boolean
	loop?: boolean
	className?: string
	/**
	 * Путь к JSON-файлу (например, '/lottie/robot-scan.json').
	 * Если не указан, используется встроенная заглушка.
	 */
	src?: string
}

// Простой контейнер Lottie. Предпочтительно использовать prop src
// (файл лежит в public), мы загрузим JSON через fetch и передадим как animationData
// — так избегаем XHR внутри lottie и ошибки с responseText.
const LottieScanner: React.FC<LottieScannerProps> = ({
	playing,
	loop = true,
	className,
	src,
}) => {
	const containerRef = useRef<HTMLDivElement | null>(null)
	const animRef = useRef<AnimationItem | null>(null)

	useEffect(() => {
		if (!containerRef.current) return

		// функция создания анимации по готовым данным
		const create = (animationData: any) => {
			animRef.current?.destroy()
			animRef.current = lottie.loadAnimation({
				container: containerRef.current!,
				renderer: 'svg',
				loop,
				autoplay: playing,
				animationData,
				rendererSettings: { progressiveLoad: true },
			})
		}

		if (src) {
			// Загружаем JSON самостоятельно и отдаём как animationData
			fetch(src)
				.then(async r => {
					// на всякий случай пробуем текст -> JSON, чтобы избежать mime/responseType сюрпризов
					const text = await r.text()
					return JSON.parse(text)
				})
				.then(data => create(data))
				.catch(err => {
					console.error('Ошибка загрузки Lottie JSON:', err)
					// fallback: встроенная заглушка
					create(getFallbackAnimation())
				})
		} else {
			create(getFallbackAnimation())
		}

		return () => {
			animRef.current?.destroy()
			animRef.current = null
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [src])

	useEffect(() => {
		if (!animRef.current) return
		if (playing) animRef.current.play()
		else animRef.current.pause()
	}, [playing])

	return <div ref={containerRef} className={className} />
}

function getFallbackAnimation() {
	return {
		v: '5.8.1',
		fr: 60,
		ip: 0,
		op: 180,
		w: 1200,
		h: 180,
		nm: 'scanner',
		ddd: 0,
		assets: [],
		layers: [
			{
				ddd: 0,
				ind: 1,
				ty: 4,
				nm: 'bg',
				sr: 1,
				sw: 1200,
				sh: 180,
				sc: '#000000',
				ks: {
					o: { a: 0, k: 100 },
					r: { a: 0, k: 0 },
					p: { a: 0, k: [600, 90, 0] },
					a: { a: 0, k: [0, 0, 0] },
					s: { a: 0, k: [100, 100, 100] },
				},
				shapes: [
					{
						ty: 'rc',
						nm: 'rect',
						d: 1,
						s: { a: 0, k: [1200, 180] },
						p: { a: 0, k: [0, 0] },
						r: { a: 0, k: 22 },
						st: { a: 0, k: 0 },
						fill: { ty: 'fl', c: { a: 0, k: [1, 1, 1, 0.08] } },
					},
				],
				ao: 0,
				ip: 0,
				op: 180,
				st: 0,
				bm: 0,
			},
			{
				ddd: 0,
				ind: 2,
				ty: 4,
				nm: 'line',
				sr: 1,
				ks: {
					o: { a: 0, k: 100 },
					r: { a: 0, k: 0 },
					p: { a: 0, k: [0, 0, 0] },
					a: { a: 0, k: [0, 0, 0] },
					s: { a: 0, k: [100, 100, 100] },
				},
				shapes: [
					{
						ty: 'sh',
						nm: 'scan',
						ks: {
							a: 0,
							k: {
								i: [],
								o: [],
								v: [
									[-600, -60],
									[600, -60],
								],
								c: false,
							},
						},
						st: {
							ty: 'st',
							c: { a: 0, k: [0.29, 0.54, 0.98, 1] },
							w: { a: 0, k: 4 },
							lc: 2,
							lj: 2,
						},
					},
				],
				ao: 0,
				ip: 0,
				op: 180,
				st: 0,
				bm: 0,
				masksProperties: [
					{
						inv: false,
						mode: 'a',
						o: { a: 0, k: 100 },
						x: {
							a: 0,
							k: {
								i: [],
								o: [],
								v: [
									[-600, -90],
									[600, -90],
									[600, 90],
									[-600, 90],
								],
								c: true,
							},
						},
					},
				],
				ef: [
					{
						ty: 5,
						nm: 'PosY',
						ef: [
							{
								nm: 'ADBE Slider Control-0001',
								v: {
									a: 1,
									k: [
										{ t: 0, s: -60 },
										{ t: 90, s: 60 },
										{ t: 180, s: -60 },
									],
								},
							},
						],
					},
				],
			},
		],
	} as any
}

export default LottieScanner
