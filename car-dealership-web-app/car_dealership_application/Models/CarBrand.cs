using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Threading.Tasks;

namespace CarDealershipWebApp.Models
{
    public class CarBrand
    {
        public int ID { get; set; }
        [Display(Name = "Brand")]
        public string Name { get; set; }
        public ICollection<CarModel> CarModels { get; set; }
    }
}
