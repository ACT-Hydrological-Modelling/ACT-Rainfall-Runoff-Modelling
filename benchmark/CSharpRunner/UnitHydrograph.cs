using System;

namespace SacramentoRunner
{
    /// <summary>
    /// A discrete convolution unit hydrograph for routing surface runoff.
    /// This is a standalone implementation matching the TIME framework's UnitHydrograph.
    /// </summary>
    public class UnitHydrograph
    {
        public double[] sCurve;
        private double[] stores;

        public UnitHydrograph()
        {
            sCurve = new double[0];
            stores = new double[0];
        }

        public UnitHydrograph(UnitHydrograph other)
        {
            if (other.sCurve != null)
            {
                sCurve = new double[other.sCurve.Length];
                Array.Copy(other.sCurve, sCurve, other.sCurve.Length);
            }
            else
            {
                sCurve = new double[0];
            }

            if (other.stores != null)
            {
                stores = new double[other.stores.Length];
                Array.Copy(other.stores, stores, other.stores.Length);
            }
            else
            {
                stores = new double[0];
            }
        }

        /// <summary>
        /// Initialize the unit hydrograph with the given proportions.
        /// </summary>
        /// <param name="proportions">Array of proportions (should sum to 1.0)</param>
        public void initialiseHydrograph(double[] proportions)
        {
            sCurve = new double[proportions.Length];
            Array.Copy(proportions, sCurve, proportions.Length);
            stores = new double[proportions.Length];
        }

        /// <summary>
        /// Process one time step of the unit hydrograph.
        /// </summary>
        /// <param name="input">Input flow to be routed</param>
        /// <returns>Output flow for this time step</returns>
        public double runTimeStep(double input)
        {
            if (sCurve == null || sCurve.Length == 0)
                return input;

            // Add input distributed across stores according to sCurve
            for (int i = 0; i < stores.Length; i++)
            {
                stores[i] += input * sCurve[i];
            }

            // Output is the first store
            double output = stores[0];

            // Shift stores (cascade down)
            for (int i = 0; i < stores.Length - 1; i++)
            {
                stores[i] = stores[i + 1];
            }
            stores[stores.Length - 1] = 0.0;

            return output;
        }

        /// <summary>
        /// Reset all stores to zero.
        /// </summary>
        public void Reset()
        {
            if (stores != null)
            {
                for (int i = 0; i < stores.Length; i++)
                {
                    stores[i] = 0.0;
                }
            }
        }

        /// <summary>
        /// Get the proportion for a specific lag index.
        /// </summary>
        /// <param name="i">Index (0-based)</param>
        /// <returns>Proportion value, or 0.0 if index is out of range</returns>
        public double proportionForItem(int i)
        {
            if (sCurve != null && i >= 0 && i < sCurve.Length)
                return sCurve[i];
            return 0.0;
        }
    }
}
