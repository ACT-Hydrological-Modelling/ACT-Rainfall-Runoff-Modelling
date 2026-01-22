using System;

namespace SacramentoRunner
{
    /// <summary>
    /// Standalone implementation of the Sacramento rainfall-runoff model.
    /// This is a direct port from the TIME framework version, with all TIME-specific
    /// dependencies removed while preserving the exact algorithm.
    /// </summary>
    public class Sacramento
    {
        #region Constants

        private const int lengthOfUnitHydrograph = 5;

        // Undocumented constants from the original Fortran implementation
        private const double PDN20 = 5.08;
        private const double PDNOR = 25.4;

        public const int Version = 1;

        #endregion

        #region Inputs

        public double Pet { get; set; }

        private double pliq; // redundant with rainfall, but keep the variable name from the original code

        public double rainfall;

        #endregion

        #region Outputs

        public double runoff;
        public double baseflow;

        #endregion

        #region Parameters

        public double Adimp;
        public double Lzfpm { get { return lzfpm; } set { lzfpm = value; } }
        private double lzfpm;

        public double Lzfsm { get { return lzfsm; } set { lzfsm = value; } }
        private double lzfsm;

        public double Lzpk { get { return lzpk; } set { lzpk = value; } }
        private double lzpk;

        public double Lzsk { get { return lzsk; } set { lzsk = value; } }
        private double lzsk;

        public double Lztwm { get { return lztwm; } set { lztwm = value; } }
        private double lztwm;

        public double Pctim;
        public double Pfree;
        public double Rexp;

        public double Rserv { get { return rserv; } set { rserv = value; } }
        private double rserv;

        public double Sarva;

        public double Side { get { return side; } set { side = value; } }
        private double side;

        public double Ssout;

        private double[] unscaledUnitHydrograph;

        public double Uzfwm;
        public double Uzk;
        public double Uztwm;
        public double Zperc;

        public double UH1
        {
            get { return unscaledUnitHydrograph[0]; }
            set { unscaledUnitHydrograph[0] = boundUnitHydrographComponent(value); }
        }

        public double UH2
        {
            get { return unscaledUnitHydrograph[1]; }
            set { unscaledUnitHydrograph[1] = boundUnitHydrographComponent(value); }
        }

        public double UH3
        {
            get { return unscaledUnitHydrograph[2]; }
            set { unscaledUnitHydrograph[2] = boundUnitHydrographComponent(value); }
        }

        public double UH4
        {
            get { return unscaledUnitHydrograph[3]; }
            set { unscaledUnitHydrograph[3] = boundUnitHydrographComponent(value); }
        }

        public double UH5
        {
            get { return unscaledUnitHydrograph[4]; }
            set { unscaledUnitHydrograph[4] = boundUnitHydrographComponent(value); }
        }

        #endregion

        #region State Variables

        public double Adimc { get; set; }
        public double Alzfpc { get; set; }
        public double Alzfpm { get; set; }
        public double Alzfsc { get; set; }
        public double Alzfsm { get; set; }
        public double ChannelFlow { get; set; }
        public double E3 { get; set; }
        public double E5 { get; set; }
        public double EvaporationChannelWater { get; set; }
        public double EvapUzfw { get; set; }
        public double EvapUztw { get; set; }
        public double Flobf { get; set; }
        public double Floin { get; set; }
        public double Flosf { get; set; }
        public double Flwbf { get; set; }
        public double Flwsf { get; set; }
        public double Lzfpc { get; set; }
        public double Lzfsc { get; set; }
        public double Lzmpd { get; set; }
        public double Lztwc { get; set; }
        public double Pbase { get; set; }
        public double Perc { get; set; }
        public double ReservedLowerZone { get; set; }
        public double Roimp { get; set; }
        public double SumLowerZoneCapacities { get; set; }
        public double Uzfwc { get; set; }
        public double Uztwc { get; set; }

        public double HydrographStore { get; set; }

        #endregion

        #region Mass Balance Variables

        double prevUztwc;
        double prevUzfwc;
        double prevLztwc;
        double prevHydrographStore;
        double prevLzfsc;
        double prevLzfpc;

        #endregion

        #region Unit Hydrograph

        private UnitHydrograph _unitHydrograph;

        private UnitHydrograph unitHydrograph
        {
            get { return _unitHydrograph; }
            set
            {
                _unitHydrograph = value;
                getUnitHydrographComponents();
            }
        }

        #endregion

        public Sacramento()
        {
            unscaledUnitHydrograph = new double[lengthOfUnitHydrograph];
            unitHydrograph = new UnitHydrograph();
            initParameters();
            updateInternalStates();
        }

        public void SetPrevVariablesForMassBalance()
        {
            prevUztwc = Uztwc;
            prevUzfwc = Uzfwc;
            prevLztwc = Lztwc;
            prevHydrographStore = HydrographStore;
            prevLzfsc = Lzfsc;
            prevLzfpc = Lzfpc;
        }

        public double MassBalance
        {
            get
            {
                double aet = EvapUztw + EvapUzfw + E3 + EvaporationChannelWater + E5;

                double deltaS = ((Uztwc - prevUztwc) + (Uzfwc - prevUzfwc) + (Lztwc - prevLztwc) +
                                  (Lzfsc - prevLzfsc) + (Lzfpc - prevLzfpc)) * (1.0 - Pctim) +
                                (HydrographStore - prevHydrographStore);
                double baseFlowLoss = ((Alzfsc - Lzfsc) + (Alzfpc - Lzfpc) + (Flobf - Flwbf)) * (1.0 - Pctim);
                double mb = pliq - aet - runoff - deltaS - Math.Min(Ssout, Flwbf + Flwsf) - baseFlowLoss;

                return mb;
            }
        }

        public void reset()
        {
            setStoresAndFluxesToZero();
            updateInternalStates();
            setUnitHydrographComponents();
            SetPrevVariablesForMassBalance();
        }

        public void initStoresFull()
        {
            Uzfwc = Uzfwm;
            Uztwc = Uztwm;
            Lztwc = Lztwm;
            Lzfsc = Lzfsm;
            Lzfpc = Lzfpm;

            updateInternalStates();
        }

        public void runTimeStep()
        {
            // For reference and traceability: some variables named a and b (sic) in the original code 
            // were used in alternance for different purposes, instead of the following four following variables
            double ratioUztw, ratioUzfw;
            double ratioLztw, ratioLzfw;
            double transfered; //! was named del in DLWC code

            int itime, ninc; //! increment limits or counters
            double addro, adj, bf, dinc, dlzp, dlzs, duz, hpl, lzair, pav, percfw, percs;
            double perctw, pinc, ratio, ratlp, ratls;

            double evapt; //  ! was evap / evapt in original subroutines

            //Store current values for mass balance calc
            SetPrevVariablesForMassBalance();

            ReservedLowerZone = Rserv * (Lzfpm + Lzfsm);

            // At this point in the Fortran implementation, there were some pan factors applied. 
            // This is not included here. A modified time series should be fed into the PET.
            evapt = Pet;

            pliq = rainfall;

            //! Determine evaporation from upper zone tension water store

            if (Uztwm > 0.0)
                EvapUztw = evapt * Uztwc / Uztwm;
            else
                EvapUztw = 0.0;

            //! Determine evaporation from upper zone free water
            if (Uztwc < EvapUztw)
            {
                EvapUztw = Uztwc;
                Uztwc = 0.0;
                //!     Determine evaporation from free water surface
                EvapUzfw = Math.Min((evapt - EvapUztw), Uzfwc);
                Uzfwc = Uzfwc - EvapUzfw;
            }
            else
            {
                Uztwc = Uztwc - EvapUztw;
                EvapUzfw = 0.0;
            }

            //     If the upper zone free water ratio exceeded the upper tension zone
            //     content ratio, then transfer the free water into tension until the ratios are equals
            if (Uztwm > 0.0)
                ratioUztw = Uztwc / Uztwm;
            else
                ratioUztw = 1.0;

            if (Uzfwm > 0.0)
                ratioUzfw = Uzfwc / Uzfwm;
            else
                ratioUzfw = 1.0;

            if (ratioUztw < ratioUzfw)
            {
                //! equivalent to the tension zone "sucking" the free water
                ratioUztw = (Uztwc + Uzfwc) / (Uztwm + Uzfwm);
                Uztwc = Uztwm * ratioUztw;
                Uzfwc = Uzfwm * ratioUztw;
            }
            //     Evaporation from Adimp (additional impervious area) and Lower zone tension water
            if (Uztwm + Lztwm > 0.0)
            {
                E3 = Math.Min((evapt - EvapUztw - EvapUzfw) * Lztwc / (Uztwm + Lztwm), Lztwc);
                E5 =
                    Math.Min(
                        EvapUztw +
                        ((evapt - EvapUztw - EvapUzfw) * (Adimc - EvapUztw - Uztwc) / (Uztwm + Lztwm)), Adimc);
            }
            else
            {
                E3 = 0.0;
                E5 = 0.0;
            }

            //     Compute the *transpiration*  loss from the lower zone tension
            Lztwc = Lztwc - E3;
            //     Adjust the impervious area store
            Adimc = Adimc - E5;
            EvapUztw = EvapUztw * (1 - Adimp - Pctim);
            EvapUzfw = EvapUzfw * (1 - Adimp - Pctim);
            E3 = E3 * (1 - Adimp - Pctim);
            E5 = E5 * Adimp;

            //     Resupply the lower zone tension with water from the lower zone
            //     free water, if more water is available there.
            if (Lztwm > 0.0)
                ratioLztw = Lztwc / Lztwm;
            else
                ratioLztw = 1.0;

            if (Alzfpm + Alzfsm - ReservedLowerZone + Lztwm > 0.0)
                ratioLzfw = (Alzfpc + Alzfsc - ReservedLowerZone + Lztwc) /
                            (Alzfpm + Alzfsm - ReservedLowerZone + Lztwm);
            else
                ratioLzfw = 1.0;

            if (ratioLztw < ratioLzfw)
            {
                transfered = (ratioLzfw - ratioLztw) * Lztwm;
                //       Transfer water from the lower zone secondary free water to lower zone
                //       tension water store
                Lztwc = Lztwc + transfered;
                Alzfsc = Alzfsc - transfered;
                if (Alzfsc < 0)
                {
                    //         Transfer primary free water if secondary free water is inadequate
                    Alzfpc = Alzfpc + Alzfsc;
                    Alzfsc = 0.0;
                }
            }

            //     Runoff from the impervious or water covered area
            Roimp = pliq * Pctim;

            //     Reduce the rain by the amount of upper zone tension water deficiency
            pav = pliq + Uztwc - Uztwm;
            if (pav < 0)
            {
                //!Fill the upper zone tension water as much as rain permits
                Adimc = Adimc + pliq;
                Uztwc = Uztwc + pliq;
                pav = 0.0;
            }
            else
            {
                Adimc = Adimc + Uztwm - Uztwc;
                Uztwc = Uztwm;
            }

            // The rest of this method is very close to the original Fortran implementation; 
            // Given the look of it I doubt I can get things to reproduce from first principle.
            if (pav <= PDN20)
            {
                adj = 1.0;
                itime = 2;
            }
            else
            {
                if (pav < PDNOR)
                {
                    //! Effective rainfall in a period is assumed to be half of the
                    //! period length for rain equal to the normal rainy period
                    adj = 0.5 * Math.Sqrt(pav / PDNOR);
                }
                else
                {
                    adj = 1.0 - 0.5 * PDNOR / pav;
                }
                itime = 1;
            }

            Flobf = 0.0;
            Flosf = 0.0;
            Floin = 0.0;

            //! Here again, being blindly faithful to original implementation
            hpl = Alzfpm / (Alzfpm + Alzfsm);
            for (int ii = itime; ii <= 2; ii++)
            {
                // using (int) Math.Floor to reproduce the fortran INT cast, even if I think (int) would do.
                ninc = (int)Math.Floor((Uzfwc * adj + pav) * 0.2) + 1;
                dinc = 1.0 / ninc;
                pinc = pav * dinc;
                dinc = dinc * adj;
                if (ninc == 1 && adj >= 1.0)
                {
                    duz = Uzk;
                    dlzp = Lzpk;
                    dlzs = Lzsk;
                }
                else
                {
                    if (Uzk < 1.0)
                    {
                        duz = 1.0 - Math.Pow((1.0 - Uzk), dinc);
                    }
                    else
                        duz = 1.0;

                    if (Lzpk < 1.0)
                        dlzp = 1.0 - Math.Pow((1.0 - Lzpk), dinc);
                    else
                        dlzp = 1.0;

                    if (Lzsk < 1.0)
                        dlzs = 1.0 - Math.Pow((1.0 - Lzsk), dinc);
                    else
                        dlzs = 1.0;
                }

                //       Drainage and percolation loop
                for (int inc = 1; inc <= ninc; inc++)
                {
                    ratio = (Adimc - Uztwc) / Lztwm;
                    addro = pinc * ratio * ratio;

                    //         Compute the baseflow from the lower zone
                    if (Alzfpc > 0.0)
                        bf = Alzfpc * dlzp;
                    else
                    {
                        Alzfpc = 0.0;
                        bf = 0.0;
                    }

                    Flobf = Flobf + bf;
                    Alzfpc = Alzfpc - bf;

                    if (Alzfsc > 0.0)
                        bf = Alzfsc * dlzs;
                    else
                    {
                        Alzfsc = 0.0;
                        bf = 0.0;
                    }

                    Alzfsc = Alzfsc - bf;
                    Flobf = Flobf + bf;

                    //         Adjust the upper zone for percolation and interflow
                    if (Uzfwc > 0.0)
                    {
                        //           Determine percolation from the upper zone free water
                        //           limited to available water and lower zone air space
                        lzair = Lztwm - Lztwc + Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                        if (lzair > 0.0)
                        {
                            Perc = (Pbase * dinc * Uzfwc) / Uzfwm;
                            Perc = Math.Min(Uzfwc,
                                             Perc *
                                             (1.0 +
                                               (Zperc *
                                                 Math.Pow(
                                                     (1.0 - (Alzfpc + Alzfsc + Lztwc) / (Alzfpm + Alzfsm + Lztwm)),
                                                     Rexp))));
                            Perc = Math.Min(lzair, Perc);
                            Uzfwc = Uzfwc - Perc;
                        }
                        else
                            Perc = 0.0;

                        //           Compute the interflow
                        transfered = duz * Uzfwc;
                        Floin = Floin + transfered;
                        Uzfwc = Uzfwc - transfered;

                        //           Distribute water to lower zone tension and free water stores
                        perctw = Math.Min(Perc * (1.0 - Pfree), Lztwm - Lztwc);
                        percfw = Perc - perctw;
                        //           Shift any excess lower zone free water percolation to the
                        //           lower zone tension water store
                        lzair = Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                        if (percfw > lzair)
                        {
                            perctw = perctw + percfw - lzair;
                            percfw = lzair;
                        }
                        Lztwc = Lztwc + perctw;

                        //           Distribute water between LZ free water supplemental and primary
                        if (percfw > 0.0)
                        {
                            ratlp = 1.0 - Alzfpc / Alzfpm;
                            ratls = 1.0 - Alzfsc / Alzfsm;
                            percs = Math.Min(Alzfsm - Alzfsc,
                                              percfw * (1.0 - hpl * (2.0 * ratlp) / (ratlp + ratls)));
                            Alzfsc = Alzfsc + percs;
                            //             Check for spill from supplemental to primary
                            if (Alzfsc > Alzfsm)
                            {
                                percs = percs - Alzfsc + Alzfsm;
                                Alzfsc = Alzfsm;
                            }
                            Alzfpc = Alzfpc + percfw - percs;
                            //             Check for spill from primary to supplemental
                            if (Alzfpc > Alzfpm)
                            {
                                Alzfsc = Alzfsc + Alzfpc - Alzfpm;
                                Alzfpc = Alzfpm;
                            }
                        }
                    }

                    //         Fill upper zone free water with tension water spill
                    if (pinc > 0.0)
                    {
                        pav = pinc;
                        if (pav - Uzfwm + Uzfwc <= 0)
                            Uzfwc = Uzfwc + pav;
                        else
                        {
                            pav = pav - Uzfwm + Uzfwc;
                            Uzfwc = Uzfwm;
                            Flosf = Flosf + pav;
                            addro = addro + pav * (1.0 - addro / pinc);
                        }
                    }
                    Adimc = Adimc + pinc - addro;
                    Roimp = Roimp + addro * Adimp;
                }
                adj = 1.0 - adj;
                pav = 0.0;
            }
            //     Compute the storage volumes, runoff components and evaporation
            //     Note evapotranspiration losses from the water surface and
            //     riparian vegetation areas are computed in stn7a
            Flosf = Flosf * (1.0 - Pctim - Adimp);
            Floin = Floin * (1.0 - Pctim - Adimp);
            Flobf = Flobf * (1.0 - Pctim - Adimp);

            //  !!!!!!!!!!!!!! end of call to stn7b 
            //  !!!!! following code to the end of the subroutine is part of stn7a

            Lzfsc = Alzfsc / (1.0 + Side);
            Lzfpc = Alzfpc / (1.0 + Side);

            //!Adjust flow for unit hydrograph

            //! Replacement / original code : using an object unitHydrograph
            doUnitHydrographRouting();

            Flwbf = Flobf / (1.0 + Side);
            if (Flwbf < 0.0)
                Flwbf = 0.0;

            // Calculate the BFI prior to losses, in order to keep 
            // this ratio in the final runoff and baseflow components.
            double totalBeforeChannelLosses = Flwbf + Flwsf;
            double ratioBaseflow = 0;
            if (totalBeforeChannelLosses > 0)
                ratioBaseflow = Flwbf / totalBeforeChannelLosses;

            //! Subtract losses from the total channel flow ( going to the subsurface discharge )
            ChannelFlow = Math.Max(0.0, (Flwbf + Flwsf - Ssout));
            //! following was e4 
            EvaporationChannelWater = Math.Min(evapt * Sarva, ChannelFlow);

            runoff = ChannelFlow - EvaporationChannelWater;
            baseflow = runoff * ratioBaseflow;

            if (Double.IsNaN(runoff))
                throw new Exception("Runoff is NaN - invalid parameter set");
        }

        protected void doUnitHydrographRouting()
        {
            Flwsf = unitHydrograph.runTimeStep(Flosf + Roimp + Floin);
            HydrographStore += (Flosf + Roimp + Floin - Flwsf);
        }

        private void initParameters()
        {
            for (int i = 0; i < unscaledUnitHydrograph.Length; i++)
                unscaledUnitHydrograph[i] = 0; // .NET does this, but lets be explicit

            unscaledUnitHydrograph[0] = 1.0;
            setUnitHydrographComponents();

            // Some relatively arbitrary (though sensible) default values
            Uztwm = 50;
            Uzfwm = 40;
            Lztwm = 130;
            Lzfpm = 60;
            Lzfsm = 25;
            Rserv = 0.3;
            Adimp = 0.0;
            Uzk = 0.3;
            Lzpk = 0.01;
            Lzsk = 0.05;
            Zperc = 40;
            Rexp = 1.0;
            Pctim = 0.01;
            Pfree = 0.06;
            Side = 0.0;
            Ssout = 0.0;
            Sarva = 0.0;
        }

        private void setStoresAndFluxesToZero()
        {
            pliq = 0.0;
            rainfall = 0.0;
            Pet = 0.0;

            Uzfwc = 0.0;
            Uztwc = 0.0;
            Lzfpc = 0.0;
            Lzfsc = 0.0;
            Lztwc = 0.0;
            Roimp = 0.0;
            Flobf = 0.0;
            Flosf = 0.0;
            Floin = 0.0;

            Flwbf = 0.0;
            Flwsf = 0.0;

            EvapUztw = 0.0;
            EvapUzfw = 0.0;
            E5 = 0.0;
            E3 = 0.0;
            ReservedLowerZone = 0.0;

            unitHydrograph.Reset();
            HydrographStore = 0.0;

            ChannelFlow = 0.0;
            EvaporationChannelWater = 0.0;
            Perc = 0.0;
        }

        private void setUnitHydrographComponents()
        {
            unitHydrograph.initialiseHydrograph(normalise(unscaledUnitHydrograph));
        }

        private void getUnitHydrographComponents()
        {
            if (unitHydrograph == null || unitHydrograph.sCurve == null || unscaledUnitHydrograph == null)
                return;

            for (int i = 0; i < lengthOfUnitHydrograph; i++)
                unscaledUnitHydrograph[i] = unitHydrograph.proportionForItem(i);

            setUnitHydrographComponents();
        }

        private double[] normalise(double[] unscaledUnitHydrograph)
        {
            var total = sum(unscaledUnitHydrograph);
            var tmp = new double[unscaledUnitHydrograph.Length];

            if (allZeros(unscaledUnitHydrograph)) // this can happen and cannot be prevented with an intuitive behavior.
            {
                tmp[0] = 1.0; // default to no effect.
                unitHydrograph.initialiseHydrograph(tmp);
                return tmp;
            }

            if (Math.Abs(total) < 1e-10)
                throw new ArgumentException(
                    "The sum of the unscaled components of the unit hydrograph is zero. This 'class' should not have let this happen - there is an issue");

            for (int i = 0; i < tmp.Length; i++)
                tmp[i] = unscaledUnitHydrograph[i] / total;
            return tmp;
        }

        private bool allZeros(double[] unscaledUnitHydrograph)
        {
            for (int i = 0; i < unscaledUnitHydrograph.Length; i++)
                if (unscaledUnitHydrograph[i] != 0)
                    return false;
            return true;
        }

        private double sum(double[] unscaledUnitHydrograph)
        {
            double result = 0;
            for (int i = 0; i < unscaledUnitHydrograph.Length; i++)
                result += unscaledUnitHydrograph[i];
            return result;
        }

        private double boundUnitHydrographComponent(double value)
        {
            return Math.Max(0, value);
        }

        public void updateInternalStates()
        {
            Alzfsm = Lzfsm * (1.0 + Side);
            Alzfpm = Lzfpm * (1.0 + Side);
            Alzfsc = Lzfsc * (1.0 + Side);
            Alzfpc = Lzfpc * (1.0 + Side);

            Pbase = Alzfsm * Lzsk + Alzfpm * Lzpk;
            Adimc = Uztwc + Lztwc;

            ReservedLowerZone = Rserv * (Lzfpm + Lzfsm);
            SumLowerZoneCapacities = Lztwm + Lzfpm + Lzfsm;
        }
    }
}
