import { useEffect, useState } from 'react'

export interface DecileStats {
  decile: number
  ann_return: number
  ann_std: number
  period_return_mean: number
  period_return_std: number
  n: number
}

export interface ForecastCalibration {
  model_id: string
  test_period: { start: string; end: string }
  horizons: Record<string, DecileStats[]>
}

interface UseForecastCalibrationResult {
  calibration: ForecastCalibration | null
  loading: boolean
  error: string | null
}

export function useForecastCalibration(modelId: string): UseForecastCalibrationResult {
  const [calibration, setCalibration] = useState<ForecastCalibration | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setLoading(true)
    setError(null)
    setCalibration(null)
    fetch(`./data/backtest/${modelId}/forecast_calibration.json`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<ForecastCalibration>
      })
      .then(data => setCalibration(data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [modelId])

  return { calibration, loading, error }
}
