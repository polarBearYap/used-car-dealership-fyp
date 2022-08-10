using CarDealershipWebApp.Data;
using CarDealershipWebApp.Models.Interfaces;
using CarDealershipWebApp.Utilities;
using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;

namespace CarDealershipWebApp.Models.ViewModels
{
    public class CarViewModel : CarBluePrint
    {
        [Display(Name = "Aspiration")]
        public int Aspiration
        {
            get
            {
                return Array.IndexOf(AspirationTypeList, car.Aspiration);
            }
            set
            {
                car.Aspiration = AspirationTypeList[value];
            }
        }
        [Display(Name = "Assembled")]
        public int Assembled
        {
            get
            {
                return Array.IndexOf(AssembledTypeList, car.Assembled);
            }
            set
            {
                car.Assembled = AssembledTypeList[value];
            }
        }
        [Display(Name = "Colour")]
        [Required(AllowEmptyStrings = false)]
        public string Colour
        {
            get
            {
                return car.Colour;
            }
            set
            {
                car.Colour = value;
            }
        }
        [Display(Name = "Direct Injection")]
        public int DirectInjection
        {
            get
            {
                return Array.IndexOf(DirectInjectionList, car.DirectInjection);
            }
            set
            {
                car.DirectInjection = DirectInjectionList[value];
            }
        }
        [Display(Name = "Doors")]
        [Range(2, 5)]
        public int Doors
        {
            get
            {
                return car.Doors;
            }
            set
            {
                car.Doors = value;
            }
        }
        [Display(Name = "Engine CC")]
        [Range(500, 10000)]
        public double EngineCC
        {
            get
            {
                return car.EngineCC;
            }
            set
            {
                car.EngineCC = value;
            }
        }
        [Display(Name = "Fuel Type")]
        public int FuelType
        {
            get
            {
                return Array.IndexOf(FuelTypeList, car.FuelType);
            }
            set
            {
                car.FuelType = FuelTypeList[value];
            }
        }
        [Display(Name = "Height (mm)")]
        [Range(1000, 3000)]
        public double HeightMM
        {
            get
            {
                return car.HeightMM;
            }
            set
            {
                car.HeightMM = value;
            }
        }
        [Display(Name = "Length (mm)")]
        [Range(1000, 35000)]
        public double LengthMM
        {
            get
            {
                return car.LengthMM;
            }
            set
            {
                car.LengthMM = value;
            }
        }
        [Display(Name = "Manufacture Year")]
        [Range(1960, 2022)]
        public int ManufactureYear
        {
            get
            {
                return car.ManufactureYear;
            }
            set
            {
                car.ManufactureYear = value;
            }
        }
        [Display(Name = "Mileage")]
        [Range(0, 100000000)]
        public double Mileage
        {
            get
            {
                return car.Mileage;
            }
            set
            {
                car.Mileage = value;
            }
        }
        [Display(Name = "Horse Power")]
        [Range(50, 1000)]
        public double PeakPowerHP
        {
            get
            {
                return car.PeakPowerHP;
            }
            set
            {
                car.PeakPowerHP = value;
            }
        }
        [Display(Name = "Peak Torque (nm)")]
        [Range(50, 5000)]
        public double PeakTorqueNM
        {
            get
            {
                return car.PeakTorqueNM;
            }
            set
            {
                car.PeakTorqueNM = value;
            }
        }

        [Column(TypeName = "decimal(10, 2)")]
        [Display(Name = "Price per Month")]
        [Range(100, 25000), DataType(DataType.Currency)]
        [LessEqualThan("Price", ErrorMessage = "Price per month must be less than or equal to price.")]
        public decimal PricePerMonth
        {
            get
            {
                return car.PricePerMonth;
            }
            set
            {
                car.PricePerMonth = value;
            }
        }

        [Display(Name = "Seat Capacity")]
        [RegularExpression("^(2|4|5|6|7|8|10|11|12|14)$", 
        ErrorMessage = "The seat capacity must be either 2, 4, 5, 6, 7, 8, 10, 11, 12, or 14 only.")]
        public int SeatCapacity
        {
            get
            {
                return car.SeatCapacity;
            }
            set
            {
                car.SeatCapacity = value;
            }
        }
        [Display(Name = "Steering Type")]
        public int SteeringType
        {
            get
            {
                return Array.IndexOf(SteeringTypeList, car.SteeringType);
            }
            set
            {
                car.SteeringType = SteeringTypeList[value];
            }
        }
        [Display(Name = "Title")]
        [Required(AllowEmptyStrings = false)]
        public string Title
        {
            get
            {
                return car.Title;
            }
            set
            {
                car.Title = value;
            }
        }
        [Display(Name = "Transmission")]
        public int Transmission
        {
            get
            {
                return Array.IndexOf(TransmissionList, car.Transmission);
            }
            set
            {
                car.Transmission = TransmissionList[value];
            }
        }
        [Display(Name = "Wheel Base (mm)")]
        [Range(1000, 5000)]
        [Required]
        public double WheelBaseMM
        {
            get
            {
                return car.WheelBaseMM;
            }
            set
            {
                car.WheelBaseMM = value;
            }
        }
        [Display(Name = "Width (mm)")]
        [Range(1000, 5000)]
        public double WidthMM
        {
            get
            {
                return car.WidthMM;
            }
            set
            {
                car.WidthMM = value;
            }
        }
        [Display(Name = "Update the Record to Price Analytics")]
        [Required]
        public int UpdateAnalytics
        {
            get
            {
                return Array.IndexOf(UpdateAnalyticsList, car.UpdateAnalytics);
            }
            set
            {
                car.UpdateAnalytics = UpdateAnalyticsList[value];
            }
        }
        [Column(TypeName = "decimal(10, 2)")]
        [Display(Name = "Price")]
        [Range(1000, 1000000), DataType(DataType.Currency)]
        [Required]
        public decimal Price
        {
            get
            {
                return car.AssignedPrice;
            }
            set
            {
                car.AssignedPrice = value;
            }
        }
        [Column(TypeName = "decimal(10, 2)")]
        [Display(Name = "Suggested Price")]
        [Range(1000, 1000000), DataType(DataType.Currency)]
        public decimal PredictedPrice
        {
            get
            {
                return car.PredictedPrice;
            }
            set
            {
                car.PredictedPrice = value;
            }
        }
        [Display(Name = "Created Time")]
        [DataType(DataType.DateTime)]
        [DisplayFormat(ApplyFormatInEditMode = false, DataFormatString = "{0:dd/MM/yyyy HH:mm}")]
        public DateTime CreatedTimestamp
        {
            get
            {
                return car.CreatedTimestamp;
            }
            set
            {
                car.CreatedTimestamp = value;
            }
        }

        [Display(Name = "Car Model")]
        public int CarModelID { get; set; }
        [Display(Name = "Car Brand")]
        public int CarBrandID { get; set; }

        public Car car;

        public Car GetCar(CarDealershipContext context)
        {
            CarModel carModel = context.CarModels.FirstOrDefault(cm => cm.ID == CarModelID);
            car.CarModel = carModel;
            if (car.CarModel != null)
            {
                CarBrand carBrand = context.CarBrands.FirstOrDefault(cb => cb.ID == CarBrandID);
                car.CarModel.CarBrand = carBrand;
            }
            return car;
        }

        public CarViewModel()
        {
            car = new Car();
        }

        public CarViewModel(Car car)
        {
            this.car = car;
            CarModelID = car.CarModelID;
            CarBrandID = car.CarModel.CarBrandID;
        }
    }
}
