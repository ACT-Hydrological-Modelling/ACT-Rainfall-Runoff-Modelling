import { Construction } from 'lucide-react'

export default function RunnerPlaceholder() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <Construction className="w-16 h-16 text-amber-400 mb-6" />
      <h1 className="text-2xl font-bold text-gray-900 mb-3">
        Experiment Runner
      </h1>
      <p className="text-gray-500 max-w-md">
        The experiment runner module is under development. It will allow you to configure 
        and run batch calibration experiments directly from the browser.
      </p>
    </div>
  )
}
