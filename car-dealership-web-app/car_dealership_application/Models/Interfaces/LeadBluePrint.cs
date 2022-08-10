using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CarDealershipWebApp.Models.Interfaces
{
    public class LeadBluePrint
    {
        public static readonly string[] DontEmailList = { "No", "Yes" };
        public enum DontEmailEnum : int
        {
            No, Yes
        }

        public static readonly string[] DontCallList = { "No", "Yes" };
        public enum DontCallEnum : int
        {
            No, Yes
        }

        public static readonly string[] ReceivedFreeCopyList = { "No", "Yes" };
        public enum ReceivedFreeCopyEnum : int
        {
            No, Yes
        }

        public static readonly string[] OccupationList = {
            "Currently Not Employed", "Student", "Business person", "Working Professional"
        };
        public enum OccupationEnum : int
        {
            Unemployed, Student, Businessman, WorkingProfessional
        }

        public static readonly string[] StatusList = {
            "Qualified", "Disqualified", "Active"
        };
        public enum StatusEnum : int
        {
            Qualified, Disqualified, Active
        }
    }
}
