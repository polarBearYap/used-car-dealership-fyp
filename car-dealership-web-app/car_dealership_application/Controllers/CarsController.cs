using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using CarDealershipWebApp.Data;
using CarDealershipWebApp.Models;
using CarDealershipWebApp.Utilities;
using CarDealershipWebApp.Models.ViewModels;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

namespace CarDealershipWebApp.Controllers
{
    public class CarsController : Controller
    {
        private readonly CarDealershipContext _context;

        public CarsController(CarDealershipContext context)
        {
            _context = context;
        }

        // GET: Cars
        public async Task<IActionResult> Index(string sortOrder, string searchString, string currentFilter, int? pageNumber, int? pageSize)
        {
            // Check if searchString has changed by user, if true, the page number is reset
            if (searchString != null)
                pageNumber = 1;
            // If searchString has not changed by user, remain the last search query
            else
                searchString = currentFilter;

            if (pageSize == null)
                pageSize = 20;
            else if (pageSize > 50)
                pageSize = 50;

            AssignViewDataParam(sortOrder, searchString, (int)pageSize);
            IQueryable<Car> cars = GetCarRecords(sortOrder, searchString);
            return View(await PaginatedList<Car>.CreateAsync(cars.AsNoTracking(), pageNumber ?? 1, (int)pageSize));
        }

        // GET: Cars/Details/5
        public async Task<IActionResult> Details(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }

            var car = await _context.Cars.Include(c => c.CarModel)
                                         .ThenInclude(cm => cm.CarBrand)
                                         .AsNoTracking()
                                         .FirstOrDefaultAsync(c => c.ID == id);
            if (car == null)
            {
                return NotFound();
            }

            CarViewModel carVM = new CarViewModel(car);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(carVM);
        }

        // GET: Cars/Create
        public IActionResult Create(string prevURL)
        {
            PopulateDropDownLists(null);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View();
        }

        // POST: Cars/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create(CarViewModel carVM, string prevURL)
        {
            Car car = carVM.GetCar(_context);
            if (ModelState.IsValid)
            {
                car.CreatedTimestamp = DateTime.Now;
                _context.Add(car);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }

            PopulateDropDownLists(car);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(carVM);
        }

        [HttpGet]
        public async Task<IActionResult> GetModels(string brand)
        {
            string method = HttpContext.Request.Method;

            string requestedWith = HttpContext.Request.Headers["X-Requested-With"];

            // Source: http://www.binaryintellect.net/articles/f0fa64f6-d381-4f6d-835a-d7eb842b6288.aspx
            if (method == "GET")
            {
                if (requestedWith == "XMLHttpRequest")
                {
                    var modelNames = await _context.CarModels.Where(cm => cm.CarBrand.Name.Equals(brand))
                                                             .Select(cm => new { id = cm.ID, name = cm.Name}).ToListAsync();
                    return Json(modelNames);
                }
            }

            // Method not allowed
            Response.StatusCode = 405;
            return View();
        }

        // https://stackoverflow.com/a/14106135
        [HttpPost]
        public JsonResult Predict([Bind("Aspiration", "Assembled", "Colour", "DirectInjection", "Doors", 
                                        "EngineCC", "FuelType", "HeightMM", "LengthMM", "ManufactureYear", 
                                        "Mileage", "PeakPowerHP", "PeakTorqueNM", "SeatCapacity", "SteeringType", 
                                        "Transmission", "WheelBaseMM", "WidthMM", "CarBrandID")]  CarViewModel carVM)
        {
            string method = HttpContext.Request.Method;

            string requestedWith = HttpContext.Request.Headers["X-Requested-With"];

            // Source: http://www.binaryintellect.net/articles/f0fa64f6-d381-4f6d-835a-d7eb842b6288.aspx
            if (!(method == "POST" && requestedWith == "XMLHttpRequest"))
            {
                HttpContext.Response.StatusCode = 405;
                return null;
            }

            string[] ignoredAttributes = { "PricePerMonth", "Price", "Title", "CarModelID" };

            if (!ModelState.IsValid)
            {
                var errorModel =
                        from x in ModelState.Keys
                        // Filter attributes that are not used in prediction
                        where ModelState[x].Errors.Count > 0 && !ignoredAttributes.Contains(x)
                        select new
                        {
                            key = x,
                            errors = ModelState[x].Errors
                                                  .Select(y => y.ErrorMessage)
                                                  .ToArray()
                        };

                var errorList = errorModel.ToList();

                if (errorList.Count > 0)
                    return new JsonResult(new { Valid = false, Data = errorModel });
            }

            Car car = carVM.GetCar(_context);

           var inputPrediction = new Dictionary<string, object> { 
               ["manufacture_year"] = car.ManufactureYear,
               ["mileage"] = car.Mileage,
               ["length_mm"] = car.LengthMM,
               ["engine_cc"] = car.EngineCC,
               ["aspiration"] = $"{car.Aspiration}",
               ["wheel_base_mm"] = car.WheelBaseMM,
               ["width_mm"] = car.WidthMM,
               ["direct_injection"] = $"{car.DirectInjection}",
               ["seat_capacity"] = car.SeatCapacity,
               ["peak_power_hp"] = car.PeakPowerHP,
               ["fuel_type"] = $"{car.FuelType}",
               ["steering_type"] = $"{car.SteeringType}",
               ["assembled"] = $"{car.Assembled}",
               ["height_mm"] = car.HeightMM,
               ["peak_torque_nm"] = car.PeakTorqueNM,
               ["doors"] = car.Doors,
               ["brand"] = $"{car.CarModel.CarBrand.Name}",
               ["colour"] = $"{car.Colour}",
               ["tranmission"] = $"{car.Transmission}"
           };

            //Save data
            return new JsonResult(new
            {
                Valid = true,
                Data = JsonConvert.SerializeObject(inputPrediction)
            });
        }

        // GET: Cars/Edit/5
        public async Task<IActionResult> Edit(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }

            Car car = await _context.Cars.Include(c => c.CarModel)
                                         .ThenInclude(cm => cm.CarBrand)
                                         .FirstOrDefaultAsync(c => c.ID == id);
            if (car == null)
            {
                return NotFound();
            }
            CarViewModel carVM = new CarViewModel(car);
            PopulateDropDownLists(car);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(carVM);
        }

        // POST: Cars/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, CarViewModel carVM, string prevURL)
        {
            Car car = carVM.GetCar(_context);
            car.ID = id;

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Entry(car).State = EntityState.Modified;
                    _context.Update(car);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!CarExists(car.ID))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }

            PopulateDropDownLists(car);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(carVM);
        }

        // GET: Cars/Delete/5
        public async Task<IActionResult> Delete(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }
            Car car = await _context.Cars.Include(c => c.CarModel)
                                         .ThenInclude(cm => cm.CarBrand)
                                         .FirstOrDefaultAsync(c => c.ID == id);
            if (car == null)
            {
                return NotFound();
            }
            CarViewModel carVM = new CarViewModel(car);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(carVM);
        }

        // POST: Cars/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            Car car = await _context.Cars.FindAsync(id);
            _context.Cars.Remove(car);
            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        public IActionResult PredictPrice()
        {
            PopulateDropDownLists(null);
            return View();
        }

        public IActionResult PriceAnalytics()
        {
            return View();
        }

        private bool CarExists(int id)
        {
            return _context.Cars.Any(e => e.ID == id);
        }

        private void PopulateDropDownLists(Car car)
        {
            string selectedAspirationType = car?.Aspiration;
            string selectedAssembledType = car?.Assembled;
            string selectedDirectInjectionType = car?.DirectInjection;
            string selectedFuelType = car?.FuelType;
            string selectedSteeringType = car?.SteeringType;
            string selectedTransmissionType = car?.Transmission;
            int? selectedBrandID = car?.CarModel?.CarBrandID;
            string selectedBrand = car?.CarModel?.CarBrand?.Name;
            int? selectedModelID = car?.CarModelID;
            string selectedModel = car?.CarModel?.Name;
            string selectedUpdateAnalytics = car?.UpdateAnalytics;

            ViewBag.AspirationItems = CreateSelectItems(Car.AspirationTypeList, selectedAspirationType);
            ViewBag.AssembledItems = CreateSelectItems(Car.AssembledTypeList, selectedAssembledType);
            ViewBag.DirectInjectionItems = CreateSelectItems(Car.DirectInjectionList, selectedDirectInjectionType);
            ViewBag.FuelTypeItems = CreateSelectItems(Car.FuelTypeList, selectedFuelType);
            ViewBag.SteeringTypeItems = CreateSelectItems(Car.SteeringTypeList, selectedSteeringType);
            ViewBag.TransmissionItems = CreateSelectItems(Car.TransmissionList, selectedTransmissionType);
            ViewBag.UpdateAnalyticsItems = CreateSelectItems(Car.UpdateAnalyticsList, selectedUpdateAnalytics);
            var brands = _context.CarBrands.AsNoTracking().Select(cb => new { id = cb.ID, name = cb.Name }).ToList();
            List<int> brandIDs = new List<int>();
            List<string> brandNames = new List<string>();
            foreach (var brand in brands)
            {
                brandIDs.Add(brand.id);
                brandNames.Add(brand.name);
            }
            ViewBag.BrandItems = CreateSelectItems(brandNames.ToArray(), selectedBrand, selectedBrandID, brandIDs);
            if (selectedBrand != null)
            {
                var models = _context.CarModels.Where(cm => cm.CarBrand.Name.Equals(selectedBrand))
                                                              .Select(cm => new { id = cm.ID, name = cm.Name }).ToList();
                List<int> modelIDs = new List<int>();
                List<string> modelNames = new List<string>();
                foreach (var model in models)
                {
                    modelIDs.Add(model.id);
                    modelNames.Add(model.name);
                }
                ViewBag.ModelItems = CreateSelectItems(modelNames.ToArray(), selectedModel, selectedModelID, modelIDs);
            }
        }

        private SelectList CreateSelectItems(string[] items, string selectedItemStr, int? selectedIndex = -1, IEnumerable<int> indexList = null)
        {
            // Assign default index if no specific selectedIndex is given
            if (selectedIndex == -1)
                selectedIndex = Array.IndexOf(items, selectedItemStr);
            (int? dataValueField, string dataTextField) selectedItem = (selectedIndex, selectedItemStr);
            // Assign default index value fields if no specific indexList is given
            IEnumerable<int> index = indexList ?? Enumerable.Range(0, items.Length);
            var selectItemsZipped = index.Zip(items).ToList();
            var selectItems = new List<object>();
            foreach (var (dataValue, dataText) in selectItemsZipped)
                selectItems.Add(new { dataValueField = dataValue, dataTextField = dataText });
            if (selectedItemStr != null)
                return new SelectList(selectItems, "dataValueField", "dataTextField", selectedItem);
            else
                return new SelectList(selectItems, "dataValueField", "dataTextField", null);
        }

        private void AssignViewDataParam(string sortOrder, string searchString, int pageSize)
        {
            ViewData["CurrentSort"] = sortOrder;
            ViewData["CurrentFilter"] = searchString;
            ViewData["CurrentPageSize"] = pageSize;
            ViewData["TitleSortParam"] = string.IsNullOrEmpty(sortOrder) ? "title_desc" : "";
            ViewData["BrandSortParam"] = sortOrder == "brand" ? "brand_desc" : "brand";
            ViewData["ModelSortParam"] = sortOrder == "model" ? "model_desc" : "model";
            ViewData["AspirationSortParam"] = sortOrder == "aspiration" ? "aspiration_desc" : "aspiration";
            ViewData["AssembledSortParam"] = sortOrder == "assembled" ? "assembled_desc" : "assembled";
            ViewData["ColourSortParam"] = sortOrder == "colour" ? "colour_desc" : "colour";
            ViewData["DirectInjectionSortParam"] = sortOrder == "direct_injection" ? "direct_injection_desc" : "direct_injection";
            ViewData["DoorsSortParam"] = sortOrder == "doors" ? "doors_desc" : "doors";
            ViewData["EngineCCSortParam"] = sortOrder == "engine_cc" ? "engine_cc_desc" : "engine_cc";
            ViewData["FuelTypeSortParam"] = sortOrder == "fuel_type" ? "fuel_type_desc" : "fuel_type";
            ViewData["HeightMMSortParam"] = sortOrder == "height_mm" ? "height_mm_desc" : "height_mm";
            ViewData["LengthMMSortParam"] = sortOrder == "length_mm" ? "length_mm_desc" : "length_mm";
            ViewData["ManufactureYearSortParam"] = sortOrder == "manufacture_year" ? "manufacture_year_desc" : "manufacture_year";
            ViewData["MileageSortParam"] = sortOrder == "mileage" ? "mileage_desc" : "mileage";
            ViewData["PeakPowerSortParam"] = sortOrder == "peak_power" ? "peak_power_desc" : "peak_power";
            ViewData["PeakTorqueSortParam"] = sortOrder == "peak_torque" ? "peak_torque_desc" : "peak_torque";
            ViewData["PricePerMonthSortParam"] = sortOrder == "price_per_month" ? "price_per_month_desc" : "price_per_month";
            ViewData["SeatCapacitySortParam"] = sortOrder == "seat_capacity" ? "seat_capacity_desc" : "seat_capacity";
            ViewData["SteeringTypeSortParam"] = sortOrder == "steering_type" ? "steering_type_desc" : "steering_type";
            ViewData["TransmissionSortParam"] = sortOrder == "transmission" ? "transmission_desc" : "transmission";
            ViewData["WheelBaseMMSortParam"] = sortOrder == "wheel_base_mm" ? "wheel_base_mm_desc" : "wheel_base_mm";
            ViewData["WidthMMSortParam"] = sortOrder == "width_mm" ? "width_mm_desc" : "width_mm";
            ViewData["UpdateAnalyticsSortParam"] = sortOrder == "update_analytics" ? "update_analytics_desc" : "update_analytics";
            ViewData["CreatedTimestampSortParam"] = sortOrder == "created_timestamp" ? "created_timestamp_desc" : "created_timestamp";
            ViewData["AssignedPriceSortParam"] = sortOrder == "assigned_price" ? "assigned_price_desc" : "assigned_price";
            ViewData["PredictedPriceSortParam"] = sortOrder == "predicted_price" ? "predicted_price_desc" : "predicted_price";
        }

        private void AssignPrevURL(string url)
        {
            if (url == null) return;
            Regex regex = new Regex(@"\/Cars\/?(?:$|\?)", RegexOptions.Compiled);
            if (regex.IsMatch(url))
                ViewData["PrevURL"] = url;
        }

        private IQueryable<Car> GetCarRecords(string sortOrder, string searchString)
        {
            var cars = from car in _context.Cars
                       select car;

            IQueryable<Car> carsQuery = cars.Include(c => c.CarModel)
                                            .ThenInclude(cm => cm.CarBrand)
                                            .AsQueryable();

            switch (sortOrder)
            {
                case "brand":
                    carsQuery = carsQuery.OrderBy(c => c.CarModel.CarBrand.Name);
                    break;
                case "model":
                    carsQuery = carsQuery.OrderBy(c => c.CarModel.Name);
                    break;
                case "aspiration":
                    carsQuery = carsQuery.OrderBy(c => c.Aspiration);
                    break;
                case "assembled":
                    carsQuery = carsQuery.OrderBy(c => c.Assembled);
                    break;
                case "colour":
                    carsQuery = carsQuery.OrderBy(c => c.Colour);
                    break;
                case "direct_injection":
                    carsQuery = carsQuery.OrderBy(c => c.DirectInjection);
                    break;
                case "doors":
                    carsQuery = carsQuery.OrderBy(c => c.Doors);
                    break;
                case "engine_cc":
                    carsQuery = carsQuery.OrderBy(c => c.EngineCC);
                    break;
                case "fuel_type":
                    carsQuery = carsQuery.OrderBy(c => c.FuelType);
                    break;
                case "height_mm":
                    carsQuery = carsQuery.OrderBy(c => c.HeightMM);
                    break;
                case "length_mm":
                    carsQuery = carsQuery.OrderBy(c => c.LengthMM);
                    break;
                case "manufacture_year":
                    carsQuery = carsQuery.OrderBy(c => c.ManufactureYear);
                    break;
                case "mileage":
                    carsQuery = carsQuery.OrderBy(c => c.Mileage);
                    break;
                case "peak_power":
                    carsQuery = carsQuery.OrderBy(c => c.PeakPowerHP);
                    break;
                case "peak_torque":
                    carsQuery = carsQuery.OrderBy(c => c.PeakTorqueNM);
                    break;
                case "price_per_month":
                    carsQuery = carsQuery.OrderBy(c => c.PricePerMonth);
                    break;
                case "seat_capacity":
                    carsQuery = carsQuery.OrderBy(c => c.SeatCapacity);
                    break;
                case "steering_type":
                    carsQuery = carsQuery.OrderBy(c => c.SteeringType);
                    break;
                case "transmission":
                    carsQuery = carsQuery.OrderBy(c => c.Transmission);
                    break;
                case "wheel_base_mm":
                    carsQuery = carsQuery.OrderBy(c => c.WheelBaseMM);
                    break;
                case "width_mm":
                    carsQuery = carsQuery.OrderBy(c => c.WidthMM);
                    break;
                case "update_analytics":
                    carsQuery = carsQuery.OrderBy(c => c.UpdateAnalytics);
                    break;
                case "created_timestamp":
                    carsQuery = carsQuery.OrderBy(c => c.CreatedTimestamp);
                    break;
                case "assigned_price":
                    carsQuery = carsQuery.OrderBy(c => c.AssignedPrice);
                    break;
                case "predicted_price":
                    carsQuery = carsQuery.OrderBy(c => c.PredictedPrice);
                    break;
                case "title_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Title);
                    break;
                case "brand_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.CarModel.CarBrand.Name);
                    break;
                case "model_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.CarModel.Name);
                    break;
                case "aspiration_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Aspiration);
                    break;
                case "assembled_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Assembled);
                    break;
                case "colour_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Colour);
                    break;
                case "direct_injection_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.DirectInjection);
                    break;
                case "doors_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Doors);
                    break;
                case "engine_cc_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.EngineCC);
                    break;
                case "fuel_type_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.FuelType);
                    break;
                case "height_mm_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.HeightMM);
                    break;
                case "length_mm_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.LengthMM);
                    break;
                case "manufacture_year_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.ManufactureYear);
                    break;
                case "mileage_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Mileage);
                    break;
                case "peak_power_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.PeakPowerHP);
                    break;
                case "peak_torque_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.PeakTorqueNM);
                    break;
                case "price_per_month_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.PricePerMonth);
                    break;
                case "seat_capacity_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.SeatCapacity);
                    break;
                case "steering_type_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.SteeringType);
                    break;
                case "transmission_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.Transmission);
                    break;
                case "wheel_base_mm_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.WheelBaseMM);
                    break;
                case "width_mm_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.WidthMM);
                    break;
                case "update_analytics_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.UpdateAnalytics);
                    break;
                case "created_timestamp_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.CreatedTimestamp);
                    break;
                case "assigned_price_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.AssignedPrice);
                    break;
                case "predicted_price_desc":
                    carsQuery = carsQuery.OrderByDescending(c => c.PredictedPrice);
                    break;
                default:
                    carsQuery = carsQuery.OrderBy(c => c.Title);
                    break;
            }

            if (!string.IsNullOrEmpty(searchString))
            {
                searchString = searchString.ToLower().Trim();

                carsQuery = carsQuery.Where(c => c.Title.ToLower().Contains(searchString)
                                              || c.CarModel.CarBrand.Name.ToLower().Contains(searchString)
                                              || c.CarModel.Name.ToLower().Contains(searchString)
                                              || c.Aspiration.ToLower().Contains(searchString)
                                              || c.Assembled.ToLower().Contains(searchString)
                                              || c.Colour.ToLower().Contains(searchString)
                                              || c.DirectInjection.ToLower().Contains(searchString)
                                              || c.Doors.ToString().ToLower().Contains(searchString)
                                              || c.EngineCC.ToString().ToLower().Contains(searchString)
                                              || c.FuelType.ToLower().Contains(searchString)
                                              || c.HeightMM.ToString().ToLower().Contains(searchString)
                                              || c.LengthMM.ToString().ToLower().Contains(searchString)
                                              || c.ManufactureYear.ToString().ToLower().Contains(searchString)
                                              || c.Mileage.ToString().ToLower().Contains(searchString)
                                              || c.PeakPowerHP.ToString().ToLower().Contains(searchString)
                                              || c.PeakTorqueNM.ToString().ToLower().Contains(searchString)
                                              || c.AssignedPrice.ToString().ToLower().Contains(searchString)
                                              || c.PricePerMonth.ToString().ToLower().Contains(searchString)
                                              || c.SeatCapacity.ToString().ToLower().Contains(searchString)
                                              || c.SteeringType.ToLower().Contains(searchString)
                                              || c.Transmission.ToLower().Contains(searchString)
                                              || c.WheelBaseMM.ToString().ToLower().Contains(searchString)
                                              || c.WidthMM.ToString().ToLower().Contains(searchString));
            }

            return carsQuery;
        }

        private List<Car> FilterCars(IOrderedEnumerable<Car> sortedCars, string searchString)
        {
            IQueryable<Car> carQuery = sortedCars.AsQueryable();

            if (!string.IsNullOrEmpty(searchString))
            {
                searchString = searchString.ToLower().Trim();

                carQuery = carQuery.Where(c => c.Title.ToLower().Contains(searchString)
                                            || c.CarModel.CarBrand.Name.ToLower().Contains(searchString)
                                            || c.CarModel.Name.ToLower().Contains(searchString)
                                            || c.Aspiration.ToLower().Contains(searchString)
                                            || c.Assembled.ToLower().Contains(searchString)
                                            || c.Colour.ToLower().Contains(searchString)
                                            || c.DirectInjection.ToLower().Contains(searchString)
                                            || c.Doors.ToString().ToLower().Contains(searchString)
                                            || c.EngineCC.ToString().ToLower().Contains(searchString)
                                            || c.FuelType.ToLower().Contains(searchString)
                                            || c.HeightMM.ToString().ToLower().Contains(searchString)
                                            || c.LengthMM.ToString().ToLower().Contains(searchString)
                                            || c.ManufactureYear.ToString().ToLower().Contains(searchString)
                                            || c.Mileage.ToString().ToLower().Contains(searchString)
                                            || c.PeakPowerHP.ToString().ToLower().Contains(searchString)
                                            || c.PeakTorqueNM.ToString().ToLower().Contains(searchString)
                                            || c.AssignedPrice.ToString().ToLower().Contains(searchString)
                                            || c.PricePerMonth.ToString().ToLower().Contains(searchString)
                                            || c.SeatCapacity.ToString().ToLower().Contains(searchString)
                                            || c.SteeringType.ToLower().Contains(searchString)
                                            || c.Transmission.ToLower().Contains(searchString)
                                            || c.WheelBaseMM.ToString().ToLower().Contains(searchString)
                                            || c.WidthMM.ToString().ToLower().Contains(searchString) 
                                            || c.CreatedTimestamp.ToString().ToLower().Contains(searchString) 
                                            || c.UpdateAnalytics.ToString().ToLower().Contains(searchString)
                                            || c.AssignedPrice.ToString().ToLower().Contains(searchString)
                                            || c.PredictedPrice.ToString().ToLower().Contains(searchString));
            }

            return carQuery.ToList();
        }
    }
}
