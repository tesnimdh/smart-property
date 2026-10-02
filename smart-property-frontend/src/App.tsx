import { useState } from 'react'
import Navbar from './components/Navbar'
import BuyerPage from './pages/BuyerPage'
import SellerPage from './pages/SellerPage'


export type Mode = 'buyer' | 'seller'

function App() {
  const [mode, setMode] = useState<Mode>('seller')

  return (
    <>
      <Navbar mode={mode} onModeChange={setMode} />
      <main style={{ maxWidth: 960, margin: '0 auto', padding: '2.5rem 1.5rem' }}>
        {mode === 'buyer'     && <BuyerPage />}
        {mode === 'seller'    && <SellerPage />}
      </main>
    </>
  )
}

export default App