import { Link } from 'react-router-dom'
import { 
  FlaskConical, 
  BarChart3, 
  Droplets,
  ArrowRight,
  Construction
} from 'lucide-react'

interface ModuleCardProps {
  title: string
  description: string
  icon: React.ElementType
  to: string
  available: boolean
  color: string
}

function ModuleCard({ title, description, icon: Icon, to, available, color }: ModuleCardProps) {
  const card = (
    <div className={`relative bg-white rounded-xl border-2 p-8 transition-all ${
      available 
        ? 'border-gray-200 hover:border-primary-300 hover:shadow-lg cursor-pointer' 
        : 'border-gray-100 opacity-60 cursor-not-allowed'
    }`}>
      {!available && (
        <div className="absolute top-4 right-4 flex items-center gap-1 text-xs font-medium text-amber-600 bg-amber-50 px-2 py-1 rounded-full">
          <Construction className="w-3 h-3" />
          Coming Soon
        </div>
      )}
      <div className={`inline-flex p-4 rounded-xl ${color} mb-5`}>
        <Icon className="w-8 h-8 text-white" />
      </div>
      <h2 className="text-xl font-bold text-gray-900 mb-2">{title}</h2>
      <p className="text-gray-500 mb-6 leading-relaxed">{description}</p>
      {available && (
        <div className="flex items-center text-primary-600 font-medium text-sm">
          Open Module
          <ArrowRight className="w-4 h-4 ml-2" />
        </div>
      )}
    </div>
  )

  if (!available) return card
  return <Link to={to}>{card}</Link>
}

export default function LandingPage() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <div className="inline-flex items-center justify-center mb-6">
          <Droplets className="w-12 h-12 text-primary-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-3">
          pyrrm Workbench
        </h1>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          Rainfall-runoff model calibration, analysis, and diagnostics
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <ModuleCard
          title="Experiment Runner"
          description="Configure and run calibration experiments. Set up models, objectives, and algorithms for single or batch calibrations."
          icon={FlaskConical}
          to="/runner"
          available={false}
          color="bg-purple-500"
        />
        <ModuleCard
          title="Batch Analysis"
          description="Load and diagnose batch calibration results. View clustermaps, diagnostic tables, and interactive comparison plots across experiments."
          icon={BarChart3}
          to="/analysis"
          available={true}
          color="bg-emerald-500"
        />
      </div>
    </div>
  )
}
