import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

interface ImageUploaderProps {
	onImageUpload: (file: File) => void
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload }) => {
	const onDrop = useCallback(
		(acceptedFiles: File[]) => {
			if (acceptedFiles.length > 0) {
				const file = acceptedFiles[0]
				onImageUpload(file)
			}
		},
		[onImageUpload]
	)

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
		},
		multiple: false,
		maxSize: 10 * 1024 * 1024, // 10MB
	})

	return (
		<div className='my-8'>
			<div
				{...getRootProps()}
				className={`dropzone ${isDragActive ? 'dropzone--active' : ''}`}
			>
				<input {...getInputProps()} />
				<div className='flex flex-col items-center gap-4'>
					{isDragActive ? (
						<p className='text-lg font-medium text-blue-600'>
							Отпустите файл здесь...
						</p>
					) : (
						<>
							<div className='text-6xl opacity-70'>📷</div>
							<h3 className='text-2xl font-semibold text-gray-800'>
								Загрузите изображение с текстом
							</h3>
							<p className='text-gray-600 text-lg'>
								Перетащите файл сюда или нажмите для выбора
							</p>
							<p className='text-sm text-gray-500 text-center leading-relaxed'>
								Поддерживаемые форматы: PNG, JPG, JPEG, GIF, BMP, WebP
								<br />
								Максимальный размер: 10MB
							</p>
						</>
					)}
				</div>
			</div>
		</div>
	)
}

export default ImageUploader
