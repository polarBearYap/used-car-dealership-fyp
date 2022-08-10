using CarDealershipWebApp.Models.Interfaces;
using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;

namespace CarDealershipWebApp.Models
{
    public class Car : CarBluePrint
    {
        public int ID { get; set; }
        public int CarModelID { get; set; }
        public CarModel CarModel { get; set; }

        [Column("Aspiration")]
        public string Aspiration
        {
            get
            {
                return _aspiration;
            }
            set
            {
                if (AspirationTypeList.Contains(value))
                    _aspiration = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _aspiration;

        [Column("Assembled")]
        public string Assembled
        {
            get
            {
                return _assembled;
            }
            set
            {
                if (AssembledTypeList.Contains(value))
                    _assembled = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _assembled;

        public string Colour {
            get
            {
                if (_colour == null) return null;
                return _colour.Equals(" -") ? null : _colour;
            }
            set {
                if (value == null)
                    _colour = null;
                else
                    _colour = (value.Equals("none")) ? " -" : value;
            } 
        }
        private string _colour;

        [Display(Name = "Direct Injection")]
        [Column("DirectInjection")]
        public string DirectInjection
        {
            get
            {
                return _directInjection;
            }
            set
            {
                if (DirectInjectionList.Contains(value))
                    _directInjection = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _directInjection;

        public int Doors { get; set; }

        [Display(Name = "Engine CC")]
        public double EngineCC { get; set; }

        [Display(Name = "Fuel Type")]
        [Column("FuelType")]
        public string FuelType
        {
            get
            {
                return _fuel;
            }
            set
            {
                if (FuelTypeList.Contains(value))
                    _fuel = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _fuel;

        [Display(Name = "Height (mm)")]
        public double HeightMM { get; set; }

        [Display(Name = "Length (mm)")]
        public double LengthMM { get; set; }

        [Display(Name = "Manufacture Year")]
        public int ManufactureYear { get; set; }

        public double Mileage { get; set; }

        [Display(Name = "Horse Power")]
        public double PeakPowerHP { get; set; }

        [Display(Name = "Peak Torque (nm)")]
        public double PeakTorqueNM { get; set; }

        [Display(Name = "Price")]
        [DataType(DataType.Currency)]
        [Column(TypeName = "decimal(10, 2)")]
        public decimal AssignedPrice { get; set; }

        [Display(Name = "Predicted Price")]
        [DataType(DataType.Currency)]
        [Column(TypeName = "decimal(10, 2)")]
        public decimal PredictedPrice { get; set; }

        [Display(Name = "Price per Month")]
        [DataType(DataType.Currency)]
        [Column(TypeName = "decimal(10, 2)")]
        public decimal PricePerMonth { get; set; }

        [Display(Name = "Seat Capacity")]
        public int SeatCapacity { get; set; }

        [Display(Name = "Steering Type")]
        [Column("SteeringType")]
        public string SteeringType
        {
            get
            {
                return _steeringType;
            }
            set
            {
                if (SteeringTypeList.Contains(value))
                    _steeringType = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _steeringType;

        public string Title { get; set; }

        [Column("Transmission")]
        public string Transmission
        {
            get
            {
                return _tranmission;
            }
            set
            {
                if (TransmissionList.Contains(value))
                    _tranmission = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _tranmission;

        [Display(Name = "Wheel Base (mm)")]
        public double WheelBaseMM { get; set; }

        [Display(Name = "Width (mm)")]
        public double WidthMM { get; set; }

        [Display(Name = "Time Created")]
        [DisplayFormat(DataFormatString = "{0:dd/MM/yyyy HH:mm}")]
        public DateTime CreatedTimestamp { get; set; }

        [Display(Name = "Update Analytics")]
        [Column("UpdateAnalytics")]
        public string UpdateAnalytics
        {
            get
            {
                return _updated_analytics;
            }
            set
            {
                if (UpdateAnalyticsList.Contains(value))
                    _updated_analytics = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _updated_analytics;
    }
}
