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

namespace CarDealershipWebApp.Controllers
{
    public class LeadsController: Controller
    {
        private readonly CarDealershipContext _context;

        public LeadsController(CarDealershipContext context)
        {
            _context = context;
        }

        // GET: Leads
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
            IQueryable<Lead> leads = GetLeadRecords(sortOrder, searchString);
            return View(await PaginatedList<Lead>.CreateAsync(leads.AsNoTracking(), pageNumber ?? 1, (int)pageSize));
        }

        // GET: Leads/Details/5
        public async Task<IActionResult> Details(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }

            var lead = await _context.Leads.AsNoTracking().FirstOrDefaultAsync(l => l.ID == id);

            if (lead == null)
            {
                return NotFound();
            }

            LeadViewModel leadVM = new LeadViewModel(lead);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(leadVM);
        }

        // GET: Leads/Create
        public IActionResult Create(string prevURL)
        {
            PopulateDropDownLists(null);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View();
        }

        // POST: Leads/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create(LeadViewModel leadVM, string prevURL)
        {
            Lead lead = leadVM.lead;
            if (ModelState.IsValid)
            {
                lead.CreatedTimestamp = DateTime.Now;
                _context.Add(lead);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }

            PopulateDropDownLists(lead);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(leadVM);
        }

        // GET: Leads/Edit/5
        public async Task<IActionResult> Edit(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }

            Lead lead = await _context.Leads.FirstOrDefaultAsync(c => c.ID == id);
            if (lead == null)
            {
                return NotFound();
            }
            LeadViewModel leadVM = new LeadViewModel(lead);
            PopulateDropDownLists(lead);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(leadVM);
        }

        // POST: Leads/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, LeadViewModel leadVM, string prevURL)
        {
            Lead lead = leadVM.lead;
            lead.ID = id;

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Entry(lead).State = EntityState.Modified;
                    _context.Update(lead);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!LeadExists(lead.ID))
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

            PopulateDropDownLists(lead);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(leadVM);
        }

        // GET: Leads/Delete/5
        public async Task<IActionResult> Delete(int? id, string prevURL)
        {
            if (id == null)
            {
                return NotFound();
            }
            Lead lead = await _context.Leads.FirstOrDefaultAsync(c => c.ID == id);

            if (lead == null)
            {
                return NotFound();
            }
            LeadViewModel leadVM = new LeadViewModel(lead);
            if (prevURL == null | prevURL == "")
                prevURL = Request.Headers["Referer"].ToString();
            AssignPrevURL(prevURL);
            return View(leadVM);
        }

        // POST: Leads/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            Lead lead = await _context.Leads.FindAsync(id);
            _context.Leads.Remove(lead);
            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        private bool LeadExists(int id)
        {
            return _context.Leads.Any(l => l.ID == id);
        }

        private void PopulateDropDownLists(Lead lead)
        {
            string selectedDontEmail = lead?.DontEmail;
            string selectedDontCall = lead?.DontCall;
            string selectedOccupation = lead?.Occupation;
            string selectedReceivedFreeCopy = lead?.ReceivedFreeCopy;
            string selectedStatus = lead?.Status;

            ViewBag.DontEmailItems = CreateSelectItems(Lead.DontEmailList, selectedDontEmail);
            ViewBag.DontCallItems = CreateSelectItems(Lead.DontCallList, selectedDontCall);
            ViewBag.OccupationItems = CreateSelectItems(Lead.OccupationList, selectedOccupation);
            ViewBag.ReceivedFreeCopyItems = CreateSelectItems(Lead.ReceivedFreeCopyList, selectedReceivedFreeCopy);
            ViewBag.StatusItems = CreateSelectItems(Lead.StatusList, selectedStatus);
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

        private void AssignPrevURL(string url)
        {
            if (url == null) return;
            Regex regex = new Regex(@"\/Leads\/?(?:$|\?)", RegexOptions.Compiled);
            if (regex.IsMatch(url))
                ViewData["PrevURL"] = url;
        }

        private IQueryable<Lead> GetLeadRecords(string sortOrder, string searchString)
        {
            var leads = from lead in _context.Leads
                        select lead;

            IQueryable<Lead> leadsQuery = leads.AsQueryable();

            switch (sortOrder)
            {
                case "name":
                    leadsQuery = leadsQuery.OrderBy(l => l.Name);
                    break;
                case "email":
                    leadsQuery = leadsQuery.OrderBy(l => l.Email);
                    break;
                case "phone_no":
                    leadsQuery = leadsQuery.OrderBy(l => l.PhoneNo);
                    break;
                case "dont_email":
                    leadsQuery = leadsQuery.OrderBy(l => l.DontEmail);
                    break;
                case "dont_call":
                    leadsQuery = leadsQuery.OrderBy(l => l.DontCall);
                    break;
                case "occupation":
                    leadsQuery = leadsQuery.OrderBy(l => l.Occupation);
                    break;
                case "received_free_copy":
                    leadsQuery = leadsQuery.OrderBy(l => l.ReceivedFreeCopy);
                    break;
                case "status":
                    leadsQuery = leadsQuery.OrderBy(l => l.Status);
                    break;
                case "avg_page_view_per_visit":
                    leadsQuery = leadsQuery.OrderBy(l => l.AvgPageViewPerVisit);
                    break;
                case "created_time_stamp":
                    leadsQuery = leadsQuery.OrderBy(l => l.CreatedTimestamp);
                    break;
                case "predicted_score":
                    leadsQuery = leadsQuery.OrderBy(l => l.PredictedScore);
                    break;
                case "total_site_visit":
                    leadsQuery = leadsQuery.OrderBy(l => l.TotalSiteVisit);
                    break;
                case "total_time_spend_on_site":
                    leadsQuery = leadsQuery.OrderBy(l => l.TotalTimeSpendOnSite);
                    break;
                case "name_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.Name);
                    break;
                case "email_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.Email);
                    break;
                case "phone_no_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.PhoneNo);
                    break;
                case "dont_email_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.DontEmail);
                    break;
                case "dont_call_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.DontCall);
                    break;
                case "occupation_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.Occupation);
                    break;
                case "received_free_copy_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.ReceivedFreeCopy);
                    break;
                case "status_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.Status);
                    break;
                case "avg_page_view_per_visit_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.AvgPageViewPerVisit);
                    break;
                case "created_time_stamp_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.CreatedTimestamp);
                    break;
                case "predicted_score_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.PredictedScore);
                    break;
                case "total_site_visit_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.TotalSiteVisit);
                    break;
                case "total_time_spend_on_site_desc":
                    leadsQuery = leadsQuery.OrderByDescending(l => l.TotalTimeSpendOnSite);
                    break;
                default:
                    leadsQuery = leadsQuery.OrderBy(l => l.Name);
                    break;
            }

            if (!string.IsNullOrEmpty(searchString))
            {
                searchString = searchString.ToLower().Trim();

                leadsQuery = leadsQuery.Where(l => l.Name.ToLower().Contains(searchString)
                                                || l.Email.ToLower().Contains(searchString)
                                                || l.PhoneNo.ToLower().Contains(searchString)
                                                || l.DontEmail.ToLower().Contains(searchString)
                                                || l.DontCall.ToLower().Contains(searchString)
                                                || l.Occupation.ToLower().Contains(searchString)
                                                || l.ReceivedFreeCopy.ToLower().Contains(searchString)
                                                || l.Status.ToLower().Contains(searchString)
                                                || l.AvgPageViewPerVisit.ToString().ToLower().Contains(searchString)
                                                || l.CreatedTimestamp.ToString().ToLower().Contains(searchString)
                                                || l.PredictedScore.ToString().ToLower().Contains(searchString)
                                                || l.TotalSiteVisit.ToString().ToLower().Contains(searchString)
                                                || l.TotalTimeSpendOnSite.ToString().ToLower().Contains(searchString));
            }

            return leadsQuery;
        }

        private void AssignViewDataParam(string sortOrder, string searchString, int pageSize)
        {
            ViewData["CurrentSort"] = sortOrder;
            ViewData["CurrentFilter"] = searchString;
            ViewData["CurrentPageSize"] = pageSize;
            ViewData["NameSortParam"] = string.IsNullOrEmpty(sortOrder) ? "name_desc" : "";
            ViewData["EmailSortParam"] = sortOrder == "email" ? "email_desc" : "email";
            ViewData["PhoneNoSortParam"] = sortOrder == "phone_no" ? "phone_no_desc" : "phone_no";
            ViewData["DontEmailSortParam"] = sortOrder == "dont_email" ? "dont_email_desc" : "dont_email";
            ViewData["DontCallSortParam"] = sortOrder == "dont_call" ? "dont_call_desc" : "dont_call";
            ViewData["OccupationSortParam"] = sortOrder == "occupation" ? "occupation_desc" : "occupation";
            ViewData["ReceivedFreeCopySortParam"] = sortOrder == "received_free_copy" ? "received_free_copy_desc" : "received_free_copy";
            ViewData["StatusSortParam"] = sortOrder == "status" ? "status_desc" : "status";
            ViewData["AvgPageViewPerVisitSortParam"] = sortOrder == "avg_page_view_per_visit" ? "avg_page_view_per_visit_desc" : "avg_page_view_per_visit";
            ViewData["CreatedTimestampSortParam"] = sortOrder == "created_time_stamp" ? "created_time_stamp_desc" : "created_time_stamp";
            ViewData["PredictedScoreSortParam"] = sortOrder == "predicted_score" ? "predicted_score_desc" : "predicted_score";
            ViewData["TotalSiteVisitSortParam"] = sortOrder == "total_site_visit" ? "total_site_visit_desc" : "total_site_visit";
            ViewData["TotalTimeSpendOnSiteSortParam"] = sortOrder == "total_time_spend_on_site" ? "total_time_spend_on_site_desc" : "total_time_spend_on_site";
        }
    }
}
