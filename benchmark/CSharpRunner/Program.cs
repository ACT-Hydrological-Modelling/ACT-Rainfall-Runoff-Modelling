using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;

namespace SacramentoRunner
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Sacramento Model C# Benchmark Runner");
            Console.WriteLine("=====================================");

            // Determine paths
            string baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
            string testDataDir = Path.Combine(baseDir, "test_data");

            Console.WriteLine($"Base directory: {baseDir}");
            Console.WriteLine($"Test data directory: {testDataDir}");

            // Load parameter sets
            string paramSetsPath = Path.Combine(testDataDir, "parameter_sets.json");
            Console.WriteLine($"\nLoading parameter sets from: {paramSetsPath}");
            
            var paramSets = LoadParameterSets(paramSetsPath);
            Console.WriteLine($"Loaded {paramSets.Count} parameter sets");

            // Define test configurations
            var testConfigs = new List<(string name, string inputFile, string paramSet, bool initFull)>
            {
                ("TC01_default", "synthetic_inputs.csv", "default", false),
                ("TC02_dry", "synthetic_inputs.csv", "dry_catchment", false),
                ("TC03_wet", "synthetic_inputs.csv", "wet_catchment", false),
                ("TC04_impervious", "synthetic_inputs.csv", "impervious", false),
                ("TC05_groundwater", "synthetic_inputs.csv", "deep_groundwater", false),
                ("TC06_uh", "synthetic_inputs.csv", "unit_hydrograph", false),
                ("TC07_zero_rain", "zero_rainfall_inputs.csv", "default", false),
                ("TC08_storm", "storm_event_inputs.csv", "default", false),
                ("TC09_full_stores", "synthetic_inputs.csv", "default", true),
                ("TC10_dry_spell", "long_dry_spell_inputs.csv", "default", false),
            };

            // Run each test configuration
            foreach (var (name, inputFile, paramSet, initFull) in testConfigs)
            {
                Console.WriteLine($"\n--- Running {name} ---");
                
                string inputPath = Path.Combine(testDataDir, inputFile);
                string outputPath = Path.Combine(testDataDir, $"csharp_output_{name}.csv");

                if (!File.Exists(inputPath))
                {
                    Console.WriteLine($"  Input file not found: {inputPath}");
                    continue;
                }

                if (!paramSets.ContainsKey(paramSet))
                {
                    Console.WriteLine($"  Parameter set not found: {paramSet}");
                    continue;
                }

                var inputs = LoadInputs(inputPath);
                Console.WriteLine($"  Loaded {inputs.Count} input records");

                var outputs = RunModel(inputs, paramSets[paramSet], initFull);
                Console.WriteLine($"  Model completed, {outputs.Count} outputs");

                SaveOutputs(outputs, outputPath);
                Console.WriteLine($"  Saved to: {outputPath}");
            }

            Console.WriteLine("\n=====================================");
            Console.WriteLine("C# Benchmark complete!");
        }

        static Dictionary<string, Dictionary<string, double>> LoadParameterSets(string path)
        {
            var result = new Dictionary<string, Dictionary<string, double>>();
            
            string json = File.ReadAllText(path);
            using JsonDocument doc = JsonDocument.Parse(json);
            
            foreach (var paramSet in doc.RootElement.EnumerateObject())
            {
                string setName = paramSet.Name;
                var paramsDict = new Dictionary<string, double>();
                
                if (paramSet.Value.TryGetProperty("params", out JsonElement paramsElement))
                {
                    foreach (var param in paramsElement.EnumerateObject())
                    {
                        paramsDict[param.Name] = param.Value.GetDouble();
                    }
                }
                
                result[setName] = paramsDict;
            }
            
            return result;
        }

        static List<(int timestep, double rainfall, double pet)> LoadInputs(string path)
        {
            var inputs = new List<(int, double, double)>();
            
            using var reader = new StreamReader(path);
            string headerLine = reader.ReadLine(); // Skip header
            
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                string[] parts = line.Split(',');
                int timestep = int.Parse(parts[0], CultureInfo.InvariantCulture);
                double rainfall = double.Parse(parts[1], CultureInfo.InvariantCulture);
                double pet = double.Parse(parts[2], CultureInfo.InvariantCulture);
                
                inputs.Add((timestep, rainfall, pet));
            }
            
            return inputs;
        }

        static List<OutputRecord> RunModel(
            List<(int timestep, double rainfall, double pet)> inputs,
            Dictionary<string, double> parameters,
            bool initStoresFull)
        {
            var model = new Sacramento();
            
            // Apply parameters
            ApplyParameters(model, parameters);
            
            // Update internal states after parameter changes
            model.updateInternalStates();
            model.reset();
            
            if (initStoresFull)
            {
                model.initStoresFull();
            }
            
            var outputs = new List<OutputRecord>();
            
            foreach (var (timestep, rainfall, pet) in inputs)
            {
                model.rainfall = rainfall;
                model.Pet = pet;
                model.runTimeStep();
                
                outputs.Add(new OutputRecord
                {
                    timestep = timestep,
                    runoff = model.runoff,
                    baseflow = model.baseflow,
                    uztwc = model.Uztwc,
                    uzfwc = model.Uzfwc,
                    lztwc = model.Lztwc,
                    lzfsc = model.Lzfsc,
                    lzfpc = model.Lzfpc,
                    mass_balance = model.MassBalance,
                    channel_flow = model.ChannelFlow,
                    evap_uztw = model.EvapUztw,
                    evap_uzfw = model.EvapUzfw,
                    e3 = model.E3,
                    e5 = model.E5,
                    adimc = model.Adimc,
                    alzfpc = model.Alzfpc,
                    alzfsc = model.Alzfsc,
                    flobf = model.Flobf,
                    flosf = model.Flosf,
                    floin = model.Floin,
                    flwbf = model.Flwbf,
                    flwsf = model.Flwsf,
                    roimp = model.Roimp,
                    perc = model.Perc
                });
            }
            
            return outputs;
        }

        static void ApplyParameters(Sacramento model, Dictionary<string, double> parameters)
        {
            foreach (var (name, value) in parameters)
            {
                switch (name.ToLower())
                {
                    case "uztwm": model.Uztwm = value; break;
                    case "uzfwm": model.Uzfwm = value; break;
                    case "lztwm": model.Lztwm = value; break;
                    case "lzfpm": model.Lzfpm = value; break;
                    case "lzfsm": model.Lzfsm = value; break;
                    case "rserv": model.Rserv = value; break;
                    case "adimp": model.Adimp = value; break;
                    case "uzk": model.Uzk = value; break;
                    case "lzpk": model.Lzpk = value; break;
                    case "lzsk": model.Lzsk = value; break;
                    case "zperc": model.Zperc = value; break;
                    case "rexp": model.Rexp = value; break;
                    case "pctim": model.Pctim = value; break;
                    case "pfree": model.Pfree = value; break;
                    case "side": model.Side = value; break;
                    case "ssout": model.Ssout = value; break;
                    case "sarva": model.Sarva = value; break;
                    case "uh1": model.UH1 = value; break;
                    case "uh2": model.UH2 = value; break;
                    case "uh3": model.UH3 = value; break;
                    case "uh4": model.UH4 = value; break;
                    case "uh5": model.UH5 = value; break;
                }
            }
        }

        static void SaveOutputs(List<OutputRecord> outputs, string path)
        {
            using var writer = new StreamWriter(path);
            
            // Write header
            writer.WriteLine("timestep,runoff,baseflow,uztwc,uzfwc,lztwc,lzfsc,lzfpc,mass_balance," +
                           "channel_flow,evap_uztw,evap_uzfw,e3,e5,adimc,alzfpc,alzfsc," +
                           "flobf,flosf,floin,flwbf,flwsf,roimp,perc");
            
            // Write data
            foreach (var o in outputs)
            {
                writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0},{1:G17},{2:G17},{3:G17},{4:G17},{5:G17},{6:G17},{7:G17},{8:G17}," +
                    "{9:G17},{10:G17},{11:G17},{12:G17},{13:G17},{14:G17},{15:G17},{16:G17}," +
                    "{17:G17},{18:G17},{19:G17},{20:G17},{21:G17},{22:G17},{23:G17}",
                    o.timestep, o.runoff, o.baseflow, o.uztwc, o.uzfwc, o.lztwc, o.lzfsc, o.lzfpc,
                    o.mass_balance, o.channel_flow, o.evap_uztw, o.evap_uzfw, o.e3, o.e5,
                    o.adimc, o.alzfpc, o.alzfsc, o.flobf, o.flosf, o.floin, o.flwbf, o.flwsf,
                    o.roimp, o.perc));
            }
        }
    }

    class OutputRecord
    {
        public int timestep;
        public double runoff;
        public double baseflow;
        public double uztwc;
        public double uzfwc;
        public double lztwc;
        public double lzfsc;
        public double lzfpc;
        public double mass_balance;
        public double channel_flow;
        public double evap_uztw;
        public double evap_uzfw;
        public double e3;
        public double e5;
        public double adimc;
        public double alzfpc;
        public double alzfsc;
        public double flobf;
        public double flosf;
        public double floin;
        public double flwbf;
        public double flwsf;
        public double roimp;
        public double perc;
    }
}
