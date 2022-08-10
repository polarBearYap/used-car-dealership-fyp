using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CarDealershipWebApp.Models.Interfaces
{
    public abstract class CarBluePrint
    {
        public static readonly string[] AspirationTypeList =
        {
            "Aspirated", "Supercharged intercooled", "Turbo intercooled",
            "Turbo supercharged intercooled", "Turbocharged",
            "Twin Turbo intercooled", "Twin-scroll", "Twin-scroll turbo", "Supercharged"
        };
        public enum AspirationType : int
        {
            Aspirated, SuperchargedIntercooled, TurboIntercooled,
            TurboSuperchargedIntercooled, Turbocharged,
            TwinTurboIntercooled, TwinScroll, TwinScrollTurbo, Supercharged
        }

        public static readonly string[] AssembledTypeList = {
            "Locally Built", "Official Import", "Parallel Import"
        };
        public enum AssembledType : int
        {
            LocallyBuilt, OfficialImport, ParallelImport
        }

        public static readonly string[] DirectInjectionList = {
            "Direct Injection", "Direct/Multi-point injection",
            "Multi-Point Injected", "Carburettor Single"
        };
        public enum DirectInjectionType : int
        {
            DirectInjection, DirectOrMultiPointInjection,
            MultiPointInjected, CarburettorSingle
        }

        public static readonly string[] FuelTypeList = {
            "Diesel", "Hybrid",
            "Petrol - Unleaded (ULP)",
            "Petrol - Leaded"
        };
        public enum FuelTypeEnum : int
        {
            Diesel,
            Hybrid,
            UnLeadedPetrol,
            LeadedPetrol
        }

        public static readonly string[] SteeringTypeList = {
            "Electronic Power Steering", "Hydraulic Power", "Rack and Pinion",
            "Recirculating Ball", "Worm and Roller"
        };
        public enum SteeringTypeEnum : int
        {
            ElectronicPowerSteering, HydraulicPower, RackAndPinion,
            RecirculatingBall, WormAndRoller
        }

        public static readonly string[] TransmissionList = {
            "Automatic", "Manual"
        };
        public enum TransmissionType : int
        {
            Automatic, Manual
        }

        public static readonly string[] UpdateAnalyticsList = { "No", "Yes" };
        public enum UpdateAnalyticsEnum : int
        {
            No, Yes
        }
    }
}
