using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using CarDealershipWebApp.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace CarDealershipWebApp.Data
{
    public class CarDealershipContext : IdentityDbContext
    {
        public CarDealershipContext (DbContextOptions<CarDealershipContext> options)
            : base(options)
        {
            
        }

        public DbSet<CarBrand> CarBrands { get; set; }
        public DbSet<CarModel> CarModels { get; set; }
        public DbSet<Car> Cars { get; set; }
        public DbSet<Lead> Leads { get; set; }
    }
}
