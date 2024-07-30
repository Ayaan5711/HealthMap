import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { ErrorPage, About, Home, UploadPage } from './pages';

const App = () => {
	return (
		<>
			<BrowserRouter>
				<Routes>
					<Route path="/" element={<Home />} />
					<Route path="/upload_img" element={<UploadPage />} />
					<Route path="/about_us" element={<About />} />
					<Route path='/*' element={<ErrorPage />} />
				</Routes>
				<ToastContainer position='top-center' />
			</BrowserRouter>
		</>
	)
}

export default App