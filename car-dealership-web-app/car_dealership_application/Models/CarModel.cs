using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;

namespace CarDealershipWebApp.Models
{
    public class CarModel
    {
        public int ID { get; set; }
        [Display(Name = "Model")]
        public string Name { get; set; }
        public int CarBrandID { get; set; }
        public CarBrand CarBrand { get; set; }
        public ICollection<Car> Cars { get; set; }
    }
}
