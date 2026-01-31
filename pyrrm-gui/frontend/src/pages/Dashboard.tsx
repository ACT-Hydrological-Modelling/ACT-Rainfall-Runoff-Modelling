import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { 
  Map, 
  FlaskConical, 
  CheckCircle2, 
  Clock, 
  XCircle,
  PlayCircle,
  Plus
} from 'lucide-react'
import { getCatchments, getExperiments } from '../services/api'
import type { ExperimentStatus } from '../types'

function StatCard({ 
  title, 
  value, 
  icon: Icon, 
  color 
}: { 
  title: string
  value: number | string
  icon: React.ElementType
  color: string 
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="mt-1 text-3xl font-semibold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: ExperimentStatus }) {
  const config = {
    draft: { color: 'bg-gray-100 text-gray-700', icon: Clock },
    queued: { color: 'bg-yellow-100 text-yellow-700', icon: Clock },
    running: { color: 'bg-blue-100 text-blue-700', icon: PlayCircle },
    completed: { color: 'bg-green-100 text-green-700', icon: CheckCircle2 },
    failed: { color: 'bg-red-100 text-red-700', icon: XCircle },
    cancelled: { color: 'bg-gray-100 text-gray-700', icon: XCircle },
  }[status] || { color: 'bg-gray-100 text-gray-700', icon: Clock }
  
  const Icon = config.icon
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.color}`}>
      <Icon className="w-3 h-3 mr-1" />
      {status}
    </span>
  )
}

export default function Dashboard() {
  const { data: catchments, isLoading: catchmentsLoading } = useQuery({
    queryKey: ['catchments'],
    queryFn: getCatchments,
  })
  
  const { data: experiments, isLoading: experimentsLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => getExperiments(),
  })
  
  const isLoading = catchmentsLoading || experimentsLoading
  
  // Calculate stats
  const totalCatchments = catchments?.length || 0
  const totalExperiments = experiments?.length || 0
  const runningExperiments = experiments?.filter(e => e.status === 'running').length || 0
  const completedExperiments = experiments?.filter(e => e.status === 'completed').length || 0
  
  // Get recent experiments
  const recentExperiments = experiments
    ?.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5) || []
  
  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 mt-1">
            Overview of your rainfall-runoff modeling workspace
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            to="/catchments"
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <Map className="w-4 h-4 mr-2" />
            View Catchments
          </Link>
          <Link
            to="/experiments/new"
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Experiment
          </Link>
        </div>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Catchments"
          value={isLoading ? '...' : totalCatchments}
          icon={Map}
          color="bg-blue-500"
        />
        <StatCard
          title="Total Experiments"
          value={isLoading ? '...' : totalExperiments}
          icon={FlaskConical}
          color="bg-purple-500"
        />
        <StatCard
          title="Running"
          value={isLoading ? '...' : runningExperiments}
          icon={PlayCircle}
          color="bg-yellow-500"
        />
        <StatCard
          title="Completed"
          value={isLoading ? '...' : completedExperiments}
          icon={CheckCircle2}
          color="bg-green-500"
        />
      </div>
      
      {/* Recent Experiments */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Experiments</h2>
        </div>
        
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : recentExperiments.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <FlaskConical className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>No experiments yet</p>
            <p className="text-sm mt-1">Create a catchment and run your first calibration</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {recentExperiments.map((exp) => (
              <Link
                key={exp.id}
                to={`/experiments/${exp.id}${exp.status === 'completed' ? '/results' : ''}`}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50"
              >
                <div>
                  <p className="font-medium text-gray-900">{exp.name}</p>
                  <p className="text-sm text-gray-500">
                    {exp.model_type.toUpperCase()} • 
                    Created {new Date(exp.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  {exp.best_objective !== null && exp.best_objective !== undefined && (
                    <span className="text-sm font-medium text-gray-700">
                      Obj: {exp.best_objective.toFixed(4)}
                    </span>
                  )}
                  <StatusBadge status={exp.status as ExperimentStatus} />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
