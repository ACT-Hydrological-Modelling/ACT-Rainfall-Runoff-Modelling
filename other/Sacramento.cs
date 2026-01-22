using System;
using Microsoft.Practices.EnterpriseLibrary.Validation.Validators;
using TIME.Core;
using TIME.Core.Metadata;
using TIME.Core.Units.Defaults;
using TIME.ManagedExtensions;
using TIME.Models.Components;
using TIME.Models.SnapshotManagement;
using TIME.Science;
using TIME.Validators;

namespace TIME.Models.RainfallRunoff.Sacramento
{
    /// <summary>
    /// An implementation of the version of the Sacramento rainfall-runoff model used 
    /// as part of IQQM by the New South Wales Department of Natural Resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This model was ported to a C# implementation in 2006-09, from an implementation 
    /// in Fortran for .NET, due to a lack of Fortran compiler for .NET 2.0 as of the aforementioned date. 
    /// The results (model outputs and state variables) were checked as reproduced, though it should 
    /// be noted that the previous Fortran implementation may have had some rounding issues. 
    /// For instance if the code stated:
    /// <code>
    ///  subroutine initParameters ( this )
    ///    class( Sacramento ) :: this
    ///      this%Uztwm = 50   !   mm, where this is declared as a real(8)
    ///      this%Uzfwm = 40   ! same unit until adimp
    /// ! etc.
    ///  end subroutine
    /// </code>
    /// then a dump of the value of Uztwm may actually show something like 49.9999999999567. 
    /// The differences on time series of model variables are however not noticeable on a graph.
    /// </para>
    /// <para>
    /// The original port consisted of a port of the fortran 90
    /// core subroutines SNT7A and STR7B in the Sacramento implementation
    /// (Geoff Podger - previously NSW Department of Land and Water Conservation as of 2002-06). 
    /// Most variable names were kept on purpose as they were in the original implementation.
    /// </para>
    /// </remarks>
    [Aka( "Sacramento" ), PreviousNames( "TIME.Models.Sacramento" )]
    public class Sacramento : RainfallRunoffModel, IRainfallRunoffSnapshotModel
    {
        #region Inputs

        [Input, Aka("PET"), CalculationUnits(CommonUnits.millimetres)]
        public double Pet { get; set; }

        private double pliq; // redundant with rainfall, but keep the variable name from the original code

        #endregion

        #region Parameters

        private const int lengthOfUnitHydrograph = 5;

        [Parameter,
         Summary( "Proportion impervious when tension requirements are met" ), DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.2, RangeBoundaryType.Inclusive)]
        public double Adimp;

        [Parameter, CalculationUnits(CommonUnits.millimetres), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth),
         Summary( "Lower zone free water primary storage capacity" ), DecimalPlaces( 0 ),
        WarningRangeUnitValidator(40.0, RangeBoundaryType.Inclusive, 600.0, RangeBoundaryType.Inclusive)]
        public double Lzfpm
        {
            get { return lzfpm; }
            set { lzfpm = value; }
        }
        private double lzfpm;

        [Parameter, Summary("Lower zone free water storage capacity"), DecimalPlaces(0),
        CalculationUnits(CommonUnits.millimetres), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth),
        WarningRangeUnitValidator(15.0, RangeBoundaryType.Inclusive, 300.0, RangeBoundaryType.Inclusive)]
        public double Lzfsm
        {
            get { return lzfsm; }
            set { lzfsm = value; }
        }
        private double lzfsm;

        //!.NCA (Lzsk,add=Parameter())
        //!.NCA (Lzsk,add=Minimum( 0.0_8 , 0.002_8 ))
        //!.NCA (Lzsk,add=Maximum( 1.0_8 , 0.3_8 ))
        //!.NCA (Lzsk,add=Summary(USTRING("Lower zone supplemental drainage rate")))
        //!.NCA (Lzsk,add=DecimalPlaces( 2 ))

        [Parameter, Summary( "Lower zone primary drainage rate" ), DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0.001, RangeBoundaryType.Inclusive, 0.015, RangeBoundaryType.Inclusive)]
        public double Lzpk
        {
            get { return lzpk; }
            set { lzpk = value; }
        }
        private double lzpk;


        [Parameter, Summary( "Lower zone supplemental drainage rate" ), DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0.03, RangeBoundaryType.Inclusive, 0.2, RangeBoundaryType.Inclusive)]
        public double Lzsk
        {
            get { return lzsk; }
            set { lzsk = value; }
        }
        private double lzsk;

        [Parameter, CalculationUnits(CommonUnits.millimetres), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth),
         Summary( "Lower zone tension water storage capacity" ), DecimalPlaces( 0 ),
        WarningRangeUnitValidator(75.0, RangeBoundaryType.Inclusive, 300.0, RangeBoundaryType.Inclusive)]
        public double Lztwm
        {
            get { return lztwm; }
            set
            {
                lztwm = value;
            }
        }
        private double lztwm;

        //!.NCA (Lzpk,add=Parameter())
        //!.NCA (Lzpk,add=Minimum( 0.0_8 , 0.0001_8 ))
        //!.NCA (Lzpk,add=Maximum( 1.0_8 , 0.3_8 ))
        //!.NCA (Lzpk,add=Summary(USTRING("Lower zone primary drainage rate")))
        //!.NCA (Lzpk,add=DecimalPlaces( 2 ))

        [Parameter, Summary( "Permanently impervious fraction of catchment" ),
         DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.05, RangeBoundaryType.Inclusive)]
        public double Pctim;

        [Parameter,
         Summary( "Proportion of percolated water going to lower zone storages" ), DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.5, RangeBoundaryType.Inclusive)]
        public double Pfree;

        [Parameter, Summary( "Exponential component for percolation" ),
         DecimalPlaces( 2 ),
        WarningRangeUnitValidator(1.4, RangeBoundaryType.Inclusive, 3.5, RangeBoundaryType.Inclusive)]
        public double Rexp;

        [Parameter( true ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.4, RangeBoundaryType.Inclusive),
         Summary( "Proportion of lower zone unavailable for transpiration" ), DecimalPlaces( 2 )]
        public double Rserv
        {
            get { return rserv; }
            set { rserv = value; }
        }
        private double rserv;

        [Parameter( true ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.1, RangeBoundaryType.Inclusive),
         Summary( "Proportion of basin covered by streams and lakes (usually a fixed parameter)" ),
         DecimalPlaces( 2 )]
        public double Sarva;

        [Parameter( true ), Summary( "Proportion loss of baseflow" ),
         DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.8, RangeBoundaryType.Inclusive)]
        public double Side
        {
            get { return side; }
            set { side = value; }
        }
        private double side;

        /// <summary>
        /// Loss in bed of river
        /// </summary>
        [Parameter(true), Summary("Volume loss in bed of river"), DecimalPlaces(2), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth), CalculationUnits(CommonUnits.millimetres),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 0.1, RangeBoundaryType.Inclusive), RangeUnitValidator(0.0)]
        public double Ssout;

        private double[] unscaledUnitHydrograph;

        [Parameter, Summary( "Upper zone free water storage capacity" ),
         CalculationUnits(CommonUnits.millimetres), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth), DecimalPlaces(0),
        WarningRangeUnitValidator(10.0, RangeBoundaryType.Inclusive, 75.0, RangeBoundaryType.Inclusive)]
        public double Uzfwm;

        [Parameter, Summary( "Interflow drainage rate" ), DecimalPlaces( 2 ),
        WarningRangeUnitValidator(0.2, RangeBoundaryType.Inclusive, 0.5, RangeBoundaryType.Inclusive)]
        public double Uzk;

        [Parameter, Summary( "Upper zone tension water storage capacity" ),
         CalculationUnits(CommonUnits.millimetres), DisplayUnit(UnitCategory.RainfallAndEvaporationDepth), DecimalPlaces(0),
        WarningRangeUnitValidator(25.0, RangeBoundaryType.Inclusive, 125.0, RangeBoundaryType.Inclusive)]
        public double Uztwm;

        [Parameter, Summary( "Percolation parameter" ), DecimalPlaces( 0 ),
        WarningRangeUnitValidator(20.0, RangeBoundaryType.Inclusive, 300.0, RangeBoundaryType.Inclusive)]
        public double Zperc;

        /// <summary>
        /// Gets/sets the first component of the unit hydrograph, 
        /// i.e. the proportion of runoff not lagged
        /// </summary>
        [Parameter,
         Summary( "First component of the unit hydrograph, i.e. the proportion of runoff not lagged" ),
         DecimalPlaces( 2 ),RecordIgnore]
        [CalculationUnits(CommonUnits.none),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        [RangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        public double UH1
        {
            get { return unscaledUnitHydrograph[ 0 ]; }
            set { unscaledUnitHydrograph[ 0 ] = boundUnitHydrographComponent( value ); }
        }

        /// <summary>
        /// Gets/sets the second component of the unit hydrograph, i.e. the proportion of runoff lagged by one time step
        /// </summary>
        [Parameter,
         Summary(
             "Second component of the unit hydrograph, i.e. the proportion of runoff lagged by one time step"
             ), DecimalPlaces( 2 ),RecordIgnore]
        [CalculationUnits(CommonUnits.none),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        [RangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        public double UH2
        {
            get { return unscaledUnitHydrograph[ 1 ]; }
            set { unscaledUnitHydrograph[ 1 ] = boundUnitHydrographComponent( value ); }
        }

        /// <summary>
        /// Gets/sets the the third component of the unit hydrograph, i.e. the proportion of runoff lagged by two time steps
        /// </summary>
        [Parameter,
         Summary(
             "Third component of the unit hydrograph, i.e. the proportion of runoff lagged by two time steps"
             ), DecimalPlaces( 2 ),RecordIgnore]
        [CalculationUnits(CommonUnits.none),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        [RangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        public double UH3
        {
            get { return unscaledUnitHydrograph[ 2 ]; }
            set { unscaledUnitHydrograph[ 2 ] = boundUnitHydrographComponent( value ); }
        }

        /// <summary>
        /// Gets/sets the fourth component of the unit hydrograph, i.e. the proportion of runoff lagged by three time steps
        /// </summary>
        [Parameter,
         Summary(
             "Fourth component of the unit hydrograph, i.e. the proportion of runoff lagged by three time steps"
             ), DecimalPlaces(2), RecordIgnore]
        [CalculationUnits(CommonUnits.none),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        [RangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        public double UH4
        {
            get { return unscaledUnitHydrograph[ 3 ]; }
            set { unscaledUnitHydrograph[ 3 ] = boundUnitHydrographComponent( value ); }
        }

        /// <summary>
        /// Gets/sets the fifth component of the unit hydrograph, i.e. the proportion of runoff lagged by four time steps
        /// </summary>
        [Parameter,
         Summary(
             "Fifth component of the unit hydrograph, i.e. the proportion of runoff lagged by four time steps"
             ), DecimalPlaces(2), RecordIgnore]
        [CalculationUnits(CommonUnits.none),
        WarningRangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        [RangeUnitValidator(0, RangeBoundaryType.Inclusive, 1.0, RangeBoundaryType.Inclusive)]
        public double UH5
        {
            get { return unscaledUnitHydrograph[ 4 ]; }
            set { unscaledUnitHydrograph[ 4 ] = boundUnitHydrographComponent( value ); }
        }

        #endregion

        #region state variables

        //! reservedLowerZone was 'saved' in original code. Existence of a reserved storage 
        //! in the lower zone triangulated with paper from Sooroshian

        #endregion

        #region Unit hydrograph

        //! The components of the unit hydrograph. Fixed in the DLWC implementation.
        //private double[] componentsUH = new double[2];

        //!the channel flow value is going through a unit hydrograph
        private UnitHydrograph _unitHydrograph;

        private UnitHydrograph unitHydrograph
        {
            get { return _unitHydrograph; }
            set
            {
                _unitHydrograph = value;
                getUnitHydrographComponents( );
            }
        }

        public double HydrographStore { get; set; }
        #endregion

        #region massBalance
        //Store current values for mass balance calc
        double prevUztwc; //Upper zone tension water current storage
        double prevUzfwc; //Upper zone free water current storage
        double prevLztwc; //Lower zone tension water current storage
        double prevHydrographStore;
        double prevLzfsc;
        double prevLzfpc;
//        private bool FirstCall = true; //temporary variable to print headings for debugging

        public void SetPrevVariablesForMassBalance()
        {
            prevUztwc = Uztwc; //Upper zone tension water current storage
            prevUzfwc = Uzfwc; //Upper zone free water current storage
            prevLztwc = Lztwc; //Lower zone tension water current storage
            prevHydrographStore = HydrographStore;
            prevLzfsc = Lzfsc;
            prevLzfpc = Lzfpc;
        }

        [Output]
        public override double MassBalance
        {
            get
            {
                double aet = EvapUztw + EvapUzfw + E3 + EvaporationChannelWater + E5;

                double deltaS = ( ( Uztwc - prevUztwc ) + ( Uzfwc - prevUzfwc ) + ( Lztwc - prevLztwc ) +
                                  ( Lzfsc - prevLzfsc ) + ( Lzfpc - prevLzfpc ) ) * ( 1.0 - Pctim ) +
                                ( HydrographStore - prevHydrographStore );
                double baseFlowLoss = ( ( Alzfsc - Lzfsc ) + ( Alzfpc - Lzfpc ) + ( Flobf - Flwbf ) ) * ( 1.0 - Pctim );
                double mb=pliq - aet - runoff - deltaS - Math.Min(Ssout, Flwbf + Flwsf) - baseFlowLoss;

                //using( StreamWriter outFile = new StreamWriter( "e:\\sacramento.txt", true ) )
                //{
                //    if( FirstCall )
                //    {
                //        outFile.WriteLine(
                //            "massBalance\t pliq\t aet\t runoff\t deltaS\t Math.Min(Ssout,flwbf + flwsf)\t baseFlowLoss\t evapUztw\t evapUzfw\t e3\t evaporationChannelWater\t e5\t Uztwc\t" +
                //            "prevUztwc\t Uzfwc\t prevUzfwc\t Lztwc\t prevLztwc\t Lzfsc\t prevLzfsc\t Lzfpc\t prevLzfpc\t HydrographStore\t" +
                //            "prevHydrographStore\t alzfsc\t Lzfsc\t alzfpc\t Lzfpc\tflobf\tflwbf)" );
                //        FirstCall = false;
                //    }
                //    outFile.WriteLine(
                //        "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}",
                //        mb, pliq, aet, runoff, deltaS, Math.Min( Ssout, flwbf + flwsf ), baseFlowLoss, evapUztw,
                //        evapUzfw, e3, evaporationChannelWater, e5, Uztwc, prevUztwc, Uzfwc, prevUzfwc, Lztwc, prevLztwc,
                //        Lzfsc, prevLzfsc, Lzfpc, prevLzfpc, HydrographStore, prevHydrographStore, alzfsc, Lzfsc, alzfpc,
                //        Lzfpc, flobf, flwbf );
                //    outFile.Close( );
                //}
                return mb;
            }
        }

        [State]
        public double Adimc { get; set; }

        [Aka("alzfpc"),Summary( "" ),State]
        public double Alzfpc { get; set; }

        [Aka("alzfpm"), Summary(""), State, RecordIgnore]
        public double Alzfpm { get; set; }

        [Aka("alzfsc"),Summary( "" ),State]
        public double Alzfsc { get; set; }

        [Aka("alzfsm"),Summary( "" ),State,RecordIgnore]
        public double Alzfsm { get; set; }

        [Aka("channel flow" ),Summary( "Channel flow" ),State]
        public double ChannelFlow { get; set; }

        [State]
        public double E3 { get; set; }

        [State]
        public double E5 { get; set; }

        [State]
        public double EvaporationChannelWater { get; set; }

        [Aka( "evaporation upper free water" ),Summary( "Evaporation from the upper layer free water store" ),State]
        public double EvapUzfw { get; set; }

        [Aka( "evaporation upper tension water" ),Summary( "Evaporation from the upper layer tension water store" ),State]
        public double EvapUztw { get; set; }

        [State]
        public double Flobf { get; set; }

        [Aka( "inter flow" ),Summary( "Interflow" ),State]
        public double Floin { get; set; }

        [State]
        public double Flosf { get; set; }

        [Aka( "flux value" ),Summary( "Baseflow flux value" ),State]
        public double Flwbf { get; set; }

        [Aka( "surface runoff" ),Summary( "Surface runoff" ),State]
        public double Flwsf { get; set; }

        [Aka("lzfpc"), Summary("Lower zone free water primary current storage"), MemoryState]
        public double Lzfpc { get; set; }

        [Aka("lzfsc"), Summary("Lower zone free water current storage"), MemoryState]
        public double Lzfsc { get; set; }

        [Aka( "lzmpd" ),Summary( "Lower zone percolation demand" ),State]
        public double Lzmpd { get; set; }

        [Aka( "lztwc" ),Summary( "Lower zone tension water current storage" ),MemoryState]
        public double Lztwc { get; set; }

        [Aka( "pbase" ),Summary( "Limiting rate of drainage" ),State]
        public double Pbase { get; set; }

        [Aka( "perc" ),Summary( "Percolation demand" ),State]
        public double Perc { get; set; }

        [State, RecordIgnore]
        public double ReservedLowerZone { get; set; }

        [Aka( "roimp" ),Summary( "Runoff from the impervious or water covered area" ),State]
        public double Roimp { get; set; }

        [State, RecordIgnore]
        public double SumLowerZoneCapacities { get; set; }

        [Aka( "uzfwc" ),Summary( "Upper zone free water current storage" ),MemoryState]
        public double Uzfwc { get; set; }

        [Aka( "uztwc" ),Summary( "Upper zone tension water current storage" ),MemoryState]
        public double Uztwc { get; set; }

        #endregion
        public Sacramento( )
        {
            unscaledUnitHydrograph = new double[lengthOfUnitHydrograph];
            unitHydrograph = new UnitHydrograph( );
            initParameters( );
            updateInternalStates( );

            ParameterSetName = "<< Default Sacramento parameters. >>";
        }

        public override void reset( )
        {
            base.reset( );
            setStoresAndFluxesToZero( );
            updateInternalStates( );
            setUnitHydrographComponents();
            SetPrevVariablesForMassBalance();
        }

        public override void initStoresFull( )
        {
            Uzfwc = Uzfwm;
            Uztwc = Uztwm;
            Lztwc = Lztwm;
            Lzfsc = Lzfsm;
            Lzfpc = Lzfpm;

            updateInternalStates( );
        }

        public override void runTimeStep( )
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
            SetPrevVariablesForMassBalance( );

            ReservedLowerZone = Rserv * ( Lzfpm + Lzfsm );

            // At this point in the Fortran implementation, there were some pan factors applied. 
            // This is not included here. A modified time series should be fed into the PET.
            evapt = Pet;

            pliq = rainfall;

            //! Determine evaporation from upper zone tension water store

            if( Uztwm > 0.0 )
                EvapUztw = evapt * Uztwc / Uztwm;
            else
                EvapUztw = 0.0;

            //! Determine evaporation from upper zone free water
            if( Uztwc < EvapUztw )
            {
                EvapUztw = Uztwc;
                Uztwc = 0.0;
                //!     Determine evaporation from free water surface
                EvapUzfw = Math.Min( ( evapt - EvapUztw ), Uzfwc );
                Uzfwc = Uzfwc - EvapUzfw;
            }
            else
            {
                Uztwc = Uztwc - EvapUztw;
                EvapUzfw = 0.0;
            }

            //     If the upper zone free water ratio exceeded the upper tension zone
            //     content ratio, then transfer the free water into tension until the ratios are equals
            if( Uztwm > 0.0 )
                ratioUztw = Uztwc / Uztwm;
            else
                ratioUztw = 1.0;

            if( Uzfwm > 0.0 )
                ratioUzfw = Uzfwc / Uzfwm;
            else
                ratioUzfw = 1.0;

            if( ratioUztw < ratioUzfw )
            {
                //! equivalent to the tension zone "sucking" the free water
                ratioUztw = ( Uztwc + Uzfwc ) / ( Uztwm + Uzfwm );
                Uztwc = Uztwm * ratioUztw;
                Uzfwc = Uzfwm * ratioUztw;
            }
            //     Evaporation from Adimp (additional impervious area) and Lower zone tension water
            if( Uztwm + Lztwm > 0.0 )
            {
                E3 = Math.Min( ( evapt - EvapUztw - EvapUzfw ) * Lztwc / ( Uztwm + Lztwm ), Lztwc );
                E5 =
                    Math.Min(
                        EvapUztw +
                        ( ( evapt - EvapUztw - EvapUzfw ) * ( Adimc - EvapUztw - Uztwc ) / ( Uztwm + Lztwm ) ), Adimc );
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
            EvapUztw = EvapUztw * ( 1 - Adimp - Pctim );
            EvapUzfw = EvapUzfw * ( 1 - Adimp - Pctim );
            E3 = E3 * ( 1 - Adimp - Pctim );
            E5 = E5 * Adimp;

            //     Resupply the lower zone tension with water from the lower zone
            //     free water, if more water is available there.
            if( Lztwm > 0.0 )
                ratioLztw = Lztwc / Lztwm;
            else
                ratioLztw = 1.0;

            if( Alzfpm + Alzfsm - ReservedLowerZone + Lztwm > 0.0 )
                ratioLzfw = ( Alzfpc + Alzfsc - ReservedLowerZone + Lztwc ) /
                            ( Alzfpm + Alzfsm - ReservedLowerZone + Lztwm );
            else
                ratioLzfw = 1.0;

            if( ratioLztw < ratioLzfw )
            {
                transfered = ( ratioLzfw - ratioLztw ) * Lztwm;
                //       Transfer water from the lower zone secondary free water to lower zone
                //       tension water store
                Lztwc = Lztwc + transfered;
                Alzfsc = Alzfsc - transfered;
                if( Alzfsc < 0 )
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
            if( pav < 0 )
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
            if( pav <= PDN20 )
            {
                adj = 1.0;
                itime = 2;
            }
            else
            {
                if( pav < PDNOR )
                {
                    //! Effective rainfall in a period is assumed to be half of the
                    //! period length for rain equal to the normal rainy period
                    adj = 0.5 * Math.Sqrt( pav / PDNOR );
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
            hpl = Alzfpm / ( Alzfpm + Alzfsm );
            for( int ii = itime; ii <= 2; ii++ )
            {
                // using (int) Math.Floor to reproduce the fortran INT cast, even if I think (int) would do.
                ninc = (int) Math.Floor( ( Uzfwc * adj + pav ) * 0.2 ) + 1;
                dinc = 1.0 / ninc;
                pinc = pav * dinc;
                dinc = dinc * adj;
                if( ninc == 1 && adj >= 1.0 )
                {
                    duz = Uzk;
                    dlzp = Lzpk;
                    dlzs = Lzsk;
                }
                else
                {
                    if( Uzk < 1.0 )
                    {
                        duz = 1.0 - eWater.Utilities.Math.Pow( ( 1.0 - Uzk ), dinc );
                    }
                    else
                        duz = 1.0;

                    if( Lzpk < 1.0 )
                        dlzp = 1.0 - eWater.Utilities.Math.Pow( ( 1.0 - Lzpk ), dinc );
                    else
                        dlzp = 1.0;

                    if( Lzsk < 1.0 )
                        dlzs = 1.0 - eWater.Utilities.Math.Pow( ( 1.0 - Lzsk ), dinc );
                    else
                        dlzs = 1.0;
                }

                //       Drainage and percolation loop
                for( int inc = 1; inc <= ninc; inc++ )
                {
                    ratio = ( Adimc - Uztwc ) / Lztwm;
                    addro = pinc * ratio * ratio;

                    //         Compute the baseflow from the lower zone
                    if( Alzfpc > 0.0 )
                        bf = Alzfpc * dlzp;
                    else
                    {
                        Alzfpc = 0.0;
                        bf = 0.0;
                    }

                    Flobf = Flobf + bf;
                    Alzfpc = Alzfpc - bf;

                    if( Alzfsc > 0.0 )
                        bf = Alzfsc * dlzs;
                    else
                    {
                        Alzfsc = 0.0;
                        bf = 0.0;
                    }

                    Alzfsc = Alzfsc - bf;
                    Flobf = Flobf + bf;

                    //         Adjust the upper zone for percolation and interflow
                    if( Uzfwc > 0.0 )
                    {
                        //           Determine percolation from the upper zone free water
                        //           limited to available water and lower zone air space
                        lzair = Lztwm - Lztwc + Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                        if( lzair > 0.0 )
                        {
                            Perc = ( Pbase * dinc * Uzfwc ) / Uzfwm;
                            Perc = Math.Min( Uzfwc,
                                             Perc *
                                             ( 1.0 +
                                               ( Zperc *
                                                 eWater.Utilities.Math.Pow(
                                                     ( 1.0 - ( Alzfpc + Alzfsc + Lztwc ) / ( Alzfpm + Alzfsm + Lztwm ) ),
                                                     Rexp ) ) ) );
                            Perc = Math.Min( lzair, Perc );
                            Uzfwc = Uzfwc - Perc;
                        }
                        else
                            Perc = 0.0;

                        //           Compute the interflow
                        transfered = duz * Uzfwc;
                        Floin = Floin + transfered;
                        Uzfwc = Uzfwc - transfered;

                        //           Distribute water to lower zone tension and free water stores
                        perctw = Math.Min( Perc * ( 1.0 - Pfree ), Lztwm - Lztwc );
                        percfw = Perc - perctw;
                        //           Shift any excess lower zone free water percolation to the
                        //           lower zone tension water store
                        lzair = Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                        if( percfw > lzair )
                        {
                            perctw = perctw + percfw - lzair;
                            percfw = lzair;
                        }
                        Lztwc = Lztwc + perctw;

                        //           Distribute water between LZ free water supplemental and primary
                        if( percfw > 0.0 )
                        {
                            ratlp = 1.0 - Alzfpc / Alzfpm;
                            ratls = 1.0 - Alzfsc / Alzfsm;
                            percs = Math.Min( Alzfsm - Alzfsc,
                                              percfw * ( 1.0 - hpl * ( 2.0 * ratlp ) / ( ratlp + ratls ) ) );
                            Alzfsc = Alzfsc + percs;
                            //             Check for spill from supplemental to primary
                            if( Alzfsc > Alzfsm )
                            {
                                percs = percs - Alzfsc + Alzfsm;
                                Alzfsc = Alzfsm;
                            }
                            Alzfpc = Alzfpc + percfw - percs;
                            //             Check for spill from primary to supplemental
                            if( Alzfpc > Alzfpm )
                            {
                                Alzfsc = Alzfsc + Alzfpc - Alzfpm;
                                Alzfpc = Alzfpm;
                            }
                        }
                    }

                    //         Fill upper zone free water with tension water spill
                    if( pinc > 0.0 )
                    {
                        pav = pinc;
                        if( pav - Uzfwm + Uzfwc <= 0 )
                            Uzfwc = Uzfwc + pav;
                        else
                        {
                            pav = pav - Uzfwm + Uzfwc;
                            Uzfwc = Uzfwm;
                            Flosf = Flosf + pav;
                            addro = addro + pav * ( 1.0 - addro / pinc );
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
            Flosf = Flosf * ( 1.0 - Pctim - Adimp );
            Floin = Floin * ( 1.0 - Pctim - Adimp );
            Flobf = Flobf * ( 1.0 - Pctim - Adimp );

            //  !!!!!!!!!!!!!! end of call to stn7b 
            //  !!!!! following code to the end of the subroutine is part of stn7a

            Lzfsc = Alzfsc / ( 1.0 + Side );
            Lzfpc = Alzfpc / ( 1.0 + Side );

            //!Adjust flow for unit hydrograph

            //! Replacement / original code : using an object unitHydrograph
            doUnitHydrographRouting( );

            Flwbf = Flobf / ( 1.0 + Side );
            if( Flwbf < 0.0 )
                Flwbf = 0.0;

            // Calculate the BFI prior to losses, in order to keep 
            // this ratio in the final runoff and baseflow components.
            double totalBeforeChannelLosses = Flwbf + Flwsf;
            double ratioBaseflow = 0;
            if( totalBeforeChannelLosses > 0 )
                ratioBaseflow = Flwbf / totalBeforeChannelLosses;

            //! Subtract losses from the total channel flow ( going to the subsurface discharge )
            ChannelFlow = Math.Max( 0.0, ( Flwbf + Flwsf - Ssout ) );
            //! following was e4 
            EvaporationChannelWater = Math.Min( evapt * Sarva, ChannelFlow );

            runoff = ChannelFlow - EvaporationChannelWater;
            baseflow = runoff * ratioBaseflow;

            if( Double.IsNaN( runoff ) )
                throw new InvalidParameterSetException( Constants.RUNOFF_IS_NAN );
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //! there is then some code for the flow routing and
            //! some IO in the original code
            //! This is outside of our scope so we leave it aside
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        }

        /// <summary>
        /// Perform the routing of the surface runoff 
        /// via UH. This method is protected to allow unit tests.
        /// </summary>
        protected void doUnitHydrographRouting( )
        {
            Flwsf = unitHydrograph.runTimeStep( Flosf + Roimp + Floin );
            HydrographStore += ( Flosf + Roimp + Floin - Flwsf);
        }

        public override object Clone( )
        {
            Sacramento result = (Sacramento) base.Clone( );
            result.unitHydrograph = new UnitHydrograph( unitHydrograph );
            return result;
        }

        private void initParameters( )
        {
            for( int i = 0; i < unscaledUnitHydrograph.Length; i++ )
                unscaledUnitHydrograph[ i ] = 0; // .NET does this, but lets be explicit
            
            unscaledUnitHydrograph[ 0 ] = 1.0;
            setUnitHydrographComponents( );

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

        private void setStoresAndFluxesToZero( )
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

            unitHydrograph.Reset( );
            HydrographStore = 0.0;

            ChannelFlow = 0.0;
            EvaporationChannelWater = 0.0;
            Perc = 0.0;
        }

        private void setUnitHydrographComponents( )
        {
            unitHydrograph.initialiseHydrograph( normalise( unscaledUnitHydrograph ) );
        }

        private void getUnitHydrographComponents( )
        {
            if( unitHydrograph == null || unitHydrograph.sCurve == null || unscaledUnitHydrograph == null )
                return;

            for( int i = 0; i < lengthOfUnitHydrograph; i++ )
                unscaledUnitHydrograph[ i ] = unitHydrograph.proportionForItem( i );

            setUnitHydrographComponents( );
        }

        private double[] normalise( double[] unscaledUnitHydrograph )
        {
            var total = sum( unscaledUnitHydrograph );
            var tmp = new double[unscaledUnitHydrograph.Length];

            if( allZeros( unscaledUnitHydrograph ) ) // this can happen and cannot be prevented with an intuitive behavior.
            {
                tmp[ 0 ] = 1.0; // default to no effect.
                unitHydrograph.initialiseHydrograph( tmp );
                return tmp;
            }
            
            if( total.EqualWithTolerance(0))
                throw new ArgumentException(
                    "The sum of the unscaled components of the unit hydrograph is zero. This 'class' should not have let this happen - there is an issue" );

            for( int i = 0; i < tmp.Length; i++ )
                tmp[ i ] = unscaledUnitHydrograph[ i ] / total;
            return tmp;
        }

        private bool allZeros( double[] unscaledUnitHydrograph )
        {
            for( int i = 0; i < unscaledUnitHydrograph.Length; i++ )
                if( unscaledUnitHydrograph[ i ] != 0 )
                    return false;
            return true;
        }

        private double sum( double[] unscaledUnitHydrograph )
        {
            double result = 0;
            for( int i = 0; i < unscaledUnitHydrograph.Length; i++ )
                result += unscaledUnitHydrograph[ i ];
            return result;
        }

        private double boundUnitHydrographComponent( double value )
        {
            return Math.Max( 0, value );
        }

        private void updateInternalStates( )
        {
            Alzfsm = Lzfsm * ( 1.0 + Side );
            Alzfpm = Lzfpm * ( 1.0 + Side );
            Alzfsc = Lzfsc * ( 1.0 + Side );
            Alzfpc = Lzfpc * ( 1.0 + Side );

            Pbase = Alzfsm * Lzsk + Alzfpm * Lzpk;
            Adimc = Uztwc + Lztwc;

            ReservedLowerZone = Rserv * ( Lzfpm + Lzfsm );
            SumLowerZoneCapacities = Lztwm + Lzfpm + Lzfsm; //! + this.reserve, but cannot find in DLWC source...
        }

        #region Undocumented constants from the original Fortran implementation

        // The following constants were used with no explanation in the 
        // original Fortran code...
        private const double PDN20 = 5.08;
        private const double PDNOR = 25.4;

        #endregion

        public Guid SnapshotEntityId { get; set; } = Guid.NewGuid();

        public string SnapshotEntityIdString
        {
            get { return SnapshotEntityId.ToString(); }
            set { if (!string.IsNullOrEmpty(value)) SnapshotEntityId = Guid.Parse(value); }
        }

        public void SetSnapshot(IRainfallRunoffSnapshot snapshot)
        {
            var sacramentoState = snapshot as SacramentoSnapshot;
            if (sacramentoState == null)
                throw new Exception($"State is incorrect type - expected: {nameof(SacramentoSnapshot)} received: {snapshot.GetType()}");

            Uztwc = sacramentoState.Uztwc;
            Uzfwc = sacramentoState.Uzfwc;
            Lztwc = sacramentoState.Lztwc;
            Adimc = sacramentoState.Adimc;
            Alzfpc = sacramentoState.Alzfpc;
            Alzfsc = sacramentoState.Alzfsc;
        }

        public StateValidationResult IsValid(IRainfallRunoffSnapshot snapshot)
        {
            var sacramentoState = snapshot as SacramentoSnapshot;
            if (sacramentoState == null)
            {
                return new StateValidationResult(false, "Sacramento", SnapshotEntityId, $"State is incorrect type - expected: {nameof(SacramentoSnapshot)} received: {snapshot.GetType()}");
            }

            if (sacramentoState.Version != Version)
            {
                return StateValidationResult.InvalidVersion("Sacramento", SnapshotEntityId,snapshot.Version, Version);
            }

            return StateValidationResult.IsValid;
        }

        public IRainfallRunoffSnapshot GetSnapshot()
        {
            return new SacramentoSnapshot(SnapshotEntityId, Version, Uztwc, Uzfwc, Lztwc, Adimc, Alzfpc, Alzfsc);
        }

        public const int Version = 1;

        public bool HasSnapshot => true;
    }

    public class InvalidParameterSetException : Exception
    {
        public InvalidParameterSetException( string message ) : base( message ) {}
    }

    public class SacramentoSnapshot : IRainfallRunoffSnapshot
    {
        public Guid SnapshotEntityId { get; set; }
        public int Version { get; set; }
        
        public double Uztwc { get; set; }
        public double Uzfwc { get; set; }
        public double Lztwc { get; set; }
        public double Adimc { get; set; }
        public double Alzfpc { get; set; }
        public double Alzfsc { get; set; }

        public SacramentoSnapshot(Guid stateEntityId, int version, double uztwc, double uzfwc, double lztwc, double adimc, double alzfpc, double alzfsc)
        {
            SnapshotEntityId = stateEntityId;
            Version = version;
            Uztwc = uztwc;
            Uzfwc = uzfwc;
            Lztwc = lztwc;
            Adimc = adimc;
            Alzfpc = alzfpc;
            Alzfsc = alzfsc;
        }
    }
}