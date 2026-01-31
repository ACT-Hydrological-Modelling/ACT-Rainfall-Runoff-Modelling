import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Map, 
  FlaskConical,
  ChevronRight,
  Droplets
} from 'lucide-react'
import clsx from 'clsx'

// Pages
import Dashboard from './pages/Dashboard'
import CatchmentList from './pages/CatchmentList'
import CatchmentDetail from './pages/CatchmentDetail'
import DatasetDetail from './pages/DatasetDetail'
import ExperimentList from './pages/ExperimentList'
import ExperimentWizard from './pages/ExperimentWizard'
import ExperimentMonitor from './pages/ExperimentMonitor'
import ResultsPage from './pages/ResultsPage'
import ComparisonPage from './pages/ComparisonPage'

// Navigation items
const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/catchments', label: 'Catchments', icon: Map },
  { path: '/experiments', label: 'Experiments', icon: FlaskConical },
]

function Sidebar() {
  const location = useLocation()
  
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-gray-200">
        <Droplets className="w-8 h-8 text-hydro" />
        <span className="ml-3 text-xl font-bold text-gray-900">pyrrm</span>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 px-4 py-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path || 
                          (item.path !== '/' && location.pathname.startsWith(item.path))
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                'flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary-50 text-primary-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              )}
            >
              <Icon className="w-5 h-5 mr-3" />
              {item.label}
            </Link>
          )
        })}
      </nav>
      
      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          pyrrm-gui v0.1.0
        </p>
      </div>
    </aside>
  )
}

function Breadcrumbs() {
  const location = useLocation()
  const paths = location.pathname.split('/').filter(Boolean)
  
  if (paths.length === 0) return null
  
  return (
    <nav className="flex items-center text-sm text-gray-500 mb-4">
      <Link to="/" className="hover:text-gray-700">Home</Link>
      {paths.map((path, index) => (
        <span key={path} className="flex items-center">
          <ChevronRight className="w-4 h-4 mx-2" />
          <Link 
            to={`/${paths.slice(0, index + 1).join('/')}`}
            className="hover:text-gray-700 capitalize"
          >
            {path.replace(/-/g, ' ')}
          </Link>
        </span>
      ))}
    </nav>
  )
}

function App() {
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          <Breadcrumbs />
          
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/catchments" element={<CatchmentList />} />
            <Route path="/catchments/:id" element={<CatchmentDetail />} />
            <Route path="/catchments/:catchmentId/datasets/:datasetId" element={<DatasetDetail />} />
            <Route path="/experiments" element={<ExperimentList />} />
            <Route path="/experiments/new" element={<ExperimentWizard />} />
            <Route path="/experiments/:id" element={<ExperimentMonitor />} />
            <Route path="/experiments/:id/results" element={<ResultsPage />} />
            <Route path="/compare" element={<ComparisonPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
